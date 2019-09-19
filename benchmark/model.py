from math import ceil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch.utils.data.dataloader import default_collate

import torch_geometric
from torch_geometric.transforms import ToDense
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import (
    GCNConv, 
    SAGEConv, 
    DenseGCNConv, 
    DenseSAGEConv, 
    JumpingKnowledge, 
    TopKPooling,
    SAGPooling,
    EdgePooling,
    dense_diff_pool, 
    global_max_pool, 
    global_add_pool, 
    global_mean_pool
)

from kplex_pool import KPlexCover, cover_pool_node, cover_pool_edge, simplify
from kplex_pool.utils import hub_promotion
from kplex_pool.data import CustomDataset, DenseDataset



class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 mode='cat', graph_sage=False, dense=False):
        super(Block, self).__init__()

        if dense:
            module = DenseSAGEConv if graph_sage else DenseGCNConv
        else:
            module = SAGEConv if graph_sage else GCNConv

        self.mode = mode
        self.graph_sage = graph_sage
        self.convs = ModuleList([
            module(in_channels if l == 0 else hidden_channels, 
                   hidden_channels if mode is not None or l < num_layers - 1 else out_channels) 
            for l in range(num_layers)
        ])

        if mode is not None:
            self.jump = JumpingKnowledge(mode, hidden_channels, num_layers)

            if mode == 'cat':
                self.lin = Linear(hidden_channels*num_layers, out_channels)
            else:
                self.lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.mode is not None:
            self.lin.reset_parameters()
        
        self.jump.reset_parameters()

    def forward(self, x, *args, **kwargs):
        xs = []

        for conv in self.convs:
            x = F.relu(conv(x, *args, **kwargs))
            xs.append(x)
        
        if self.mode is not None:
            x = self.lin(self.jump(xs))
        
        return x


class CoverPool(torch.nn.Module):
    def __init__(self, dataset, cover_fun, hidden,
                 num_layers=2, 
                 num_inner_layers=2, 
                 dropout=0.3,
                 readout=True, 
                 graph_sage=False,
                 dense=False, 
                 normalize=False, 
                 jumping_knowledge='cat',
                 global_pool_op='add', 
                 node_pool_op='add'):
        super(CoverPool, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.normalize = normalize
        self.graph_sage = graph_sage
        self.dense = dense
        self.readout = readout
        self.dataset = dataset
        self.global_pool_op = global_add_pool if 'add' else global_mean_pool
        self.node_pool_op = node_pool_op
        self.dropout = dropout
        self.cover_fun = cover_fun

        self.conv_in = Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, dense)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(Block(2 * hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, dense))

        out_dim = 2 * num_layers * hidden if readout else 2 * hidden
        self.bn = BatchNorm1d(out_dim)
        self.lin1 = Linear(out_dim, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_in.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.bn.reset_parameters()            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def conv_forward(self, conv, x, data):
        if self.dense:
            return F.relu(conv(x, adj=data.adj))

        if self.graph_sage: 
            return F.relu(conv(x, edge_index=data.edge_index))
        
        return F.relu(conv(x, edge_index=data.edge_index, edge_weight=data.edge_attr))

    def forward(self, index):
        hierarchy = iter(self.cover_fun(self.dataset, index.view(-1).to(self.device)))
        
        if self.dense:
            hierarchy = map(lambda data: data.to(self.device), hierarchy)
        else:
            hierarchy = map(self.collate, hierarchy)
        
        data = next(hierarchy)
        batch_size = len(index)
        cover_index = data.cover_index

        x = self.conv_forward(self.conv_in, data.x, data)
        
        if self.dense:
            xs = [x.max(dim=1)[0], x.sum(dim=1)]
        else:
            xs = [ 
                self.global_pool_op(x, data.batch, batch_size), 
                global_max_pool(x, data.batch, batch_size)
            ]

        for block, data in zip(self.blocks, hierarchy):  
            x_mean = cover_pool_node(cover_index, x, data.num_nodes, pool=self.node_pool_op, dense=self.dense)
            x_max = cover_pool_node(cover_index, x, data.num_nodes, pool='max', dense=self.dense)
            x = torch.cat([x_mean, x_max], dim=-1)

            if 'cover_index' in data:
                cover_index = data.cover_index

            if self.normalize:
                x = F.normalize(x)

            x = self.conv_forward(block, x, data)

            if self.dense:
                xs.extend([x.max(dim=1)[0], x.sum(dim=1)])
            else:
                xs.append(self.global_pool_op(x, data.batch, batch_size))
                xs.append(global_max_pool(x, data.batch, batch_size))
        
        if not self.readout:
            xs = xs[-2:]
        
        x = torch.cat(xs, dim=1)
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

# Other models used for comparison, slightly modified from
# https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/kernel

class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.25, dropout=0.3,
                 num_inner_layers=2, jumping_knowledge='cat', graph_sage=False):
        super(DiffPool, self).__init__()

        num_nodes = max([data.num_nodes for data in dataset])
        self.dataset = DenseDataset(dataset)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = dropout

        if isinstance(ratio, list):
            num_layers = len(ratio) + 1
            self.ratios = ratio
        else:
            self.ratios = [ratio for _ in range(1, num_layers)]

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        self.embed_blocks.append(Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, True))
        self.pool_blocks.append(Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, True))

        for r in self.ratios:
            num_nodes = ceil(r * float(num_nodes))
            self.embed_blocks.append(Block(hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, True))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes, num_inner_layers, jumping_knowledge, graph_sage, True))

        self.jump = JumpingKnowledge(mode='cat')
        self.bn = BatchNorm1d(num_layers * hidden * 2)
        self.lin1 = Linear(num_layers * hidden * 2, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
            
        self.jump.reset_parameters()
        self.bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, index):
        data = self.dataset[index.view(-1)].to(self.device)     

        x, adj, mask = data.x, data.adj, data.mask
        
        s = self.pool_blocks[0](x, adj, mask=mask, add_loop=True)
        x = F.relu(self.embed_blocks[0](x, adj, mask=mask, add_loop=True))
        xs = [x.sum(dim=1), x.max(dim=1)[0]]
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks[1:], self.pool_blocks[1:])):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            xs.extend([x.sum(dim=1), x.max(dim=1)[0]])

            if i < len(self.embed_blocks) - 1:
                x, adj, l, e = dense_diff_pool(x, adj, s)
                link_loss += l
                ent_loss += e

        x = self.jump(xs)
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.softmax(x, dim=-1), link_loss, ent_loss

    def __repr__(self):
        return self.__class__.__name__


class PoolLoss(torch.nn.Module):
    def __init__(self, link_weight=1., ent_weight=1., *args, **kwargs):
        super(PoolLoss, self).__init__()
        self.loss = torch.nn.modules.loss.NLLLoss(*args, **kwargs)
        self.link_weight = link_weight
        self.ent_weight = ent_weight

    def forward(self, input, target):
        output, link_loss, ent_loss = input
        output = torch.log(output)

        return self.loss.forward(output, target) \
            + self.link_weight*link_loss \
            + self.ent_weight*ent_loss


class TopKPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, dropout=0.3,
                 num_inner_layers=2, jumping_knowledge='cat', graph_sage=False):
        super(TopKPool, self).__init__()
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.graph_sage = graph_sage
        self.dropout = dropout

        self.convs = torch.nn.ModuleList([
            Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
        ])

        self.convs.extend([
            Block(hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
            for i in range(num_layers - 1)
        ])

        if isinstance(ratio, list):
            num_layers = len(ratio) + 1
            self.ratios = ratio
        else:
            self.ratios = [ratio for _ in range(1, num_layers)]

        self.pools = torch.nn.ModuleList([
            TopKPooling(hidden, r) for r in self.ratios
        ])

        self.jump = JumpingKnowledge(mode='cat')
        self.bn = BatchNorm1d(num_layers * hidden * 2)
        self.lin1 = Linear(num_layers * hidden * 2, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for pool in self.pools:
            pool.reset_parameters()

        self.jump.reset_parameters()
        self.bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def forward(self, index):
        data = self.collate(self.dataset[index])

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.graph_sage:
            x = F.relu(self.convs[0](x, edge_index))
        else:
            x = F.relu(self.convs[0](x, edge_index, edge_attr))

        xs = [global_add_pool(x, batch), global_max_pool(x, batch)]

        for conv, pool in zip(self.convs[1:], self.pools):
            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr, batch=batch)

            if self.graph_sage:
                x = F.relu(conv(x, edge_index))
            else:
                x = F.relu(conv(x, edge_index, edge_attr))

            xs.extend([global_add_pool(x, batch), global_max_pool(x, batch)])

        x = self.jump(xs)
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SAGPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.5, dropout=0.3, gnn='GCNConv',
                 num_inner_layers=2, jumping_knowledge='cat', graph_sage=False):
        super(SAGPool, self).__init__()
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.graph_sage = graph_sage
        self.dropout = dropout

        self.convs = torch.nn.ModuleList([
            Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
        ])

        self.convs.extend([
            Block(hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
            for i in range(num_layers - 1)
        ])

        if isinstance(ratio, list):
            num_layers = len(ratio) + 1
            self.ratios = ratio
        else:
            self.ratios = [ratio for _ in range(1, num_layers)]

        self.pools = torch.nn.ModuleList([
            SAGPooling(hidden, r, GNN=getattr(torch_geometric.nn, gnn)) for r in self.ratios
        ])

        self.jump = JumpingKnowledge(mode='cat')
        self.bn = BatchNorm1d(num_layers * hidden * 2)
        self.lin1 = Linear(num_layers * hidden * 2, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for pool in self.pools:
            pool.reset_parameters()

        self.jump.reset_parameters()
        self.bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def forward(self, index):
        data = self.collate(self.dataset[index])

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.graph_sage:
            x = F.relu(self.convs[0](x, edge_index))
        else:
            x = F.relu(self.convs[0](x, edge_index, edge_attr))

        xs = [global_add_pool(x, batch), global_max_pool(x, batch)]

        for conv, pool in zip(self.convs[1:], self.pools):
            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr, batch=batch)

            if self.graph_sage:
                x = F.relu(conv(x, edge_index))
            else:
                x = F.relu(conv(x, edge_index, edge_attr))

            xs.extend([global_add_pool(x, batch), global_max_pool(x, batch)])

        x = self.jump(xs)
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class EdgePool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, dropout=0.3, method='softmax', edge_dropout=0.,
                 num_inner_layers=2, jumping_knowledge='cat', graph_sage=False):
        super(EdgePool, self).__init__()
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.graph_sage = graph_sage
        self.dropout = dropout

        self.convs = torch.nn.ModuleList([
            Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
        ])

        self.convs.extend([
            Block(hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
            for i in range(num_layers - 1)
        ])

        self.pools = torch.nn.ModuleList([
            EdgePooling(hidden, getattr(EdgePooling, 'compute_edge_score_' + method), edge_dropout, 0.) for i in range(num_layers - 1)
        ])

        self.jump = JumpingKnowledge(mode='cat')
        self.bn = BatchNorm1d(num_layers * hidden * 2)
        self.lin1 = Linear(num_layers * hidden * 2, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for pool in self.pools:
            pool.reset_parameters()

        self.jump.reset_parameters()
        self.bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def forward(self, index):
        data = self.collate(self.dataset[index])

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.convs[0](x, edge_index))
        xs = [global_add_pool(x, batch), global_max_pool(x, batch)]

        for conv, pool in zip(self.convs[1:], self.pools):
            x, edge_index, batch, _ = pool(x, edge_index, batch)
            x = F.relu(conv(x, edge_index))
            xs.extend([global_add_pool(x, batch), global_max_pool(x, batch)])

        x = self.jump(xs)
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
