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
from kplex_pool.data import CustomDataset



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

class KPlexPool(torch.nn.Module):
    def __init__(self, dataset, hidden, k, 
                 k_step_factor=1, 
                 num_layers=2, 
                 num_inner_layers=2, 
                 dropout=0.3,
                 readout=True, 
                 graph_sage=False, 
                 normalize=False, 
                 cache=True, 
                 jumping_knowledge='cat',
                 global_pool_op='add', 
                 node_pool_op='add', 
                 cover_priority='default',
                 kplex_priority='default',
                 skip_covered=False,
                 **cover_args):
        super(KPlexPool, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.normalize = normalize
        self.graph_sage = graph_sage
        self.readout = readout
        self.dataset = dataset
        self.global_pool_op = global_add_pool if 'add' else global_mean_pool
        self.node_pool_op = node_pool_op
        self.dropout = dropout
        self.kplex_cover = KPlexCover(cover_priority, kplex_priority, skip_covered)
        self.cover_args = cover_args

        if isinstance(k, list):
            self.ks = k
            num_layers = len(k) + 1
        else:
            self.ks = [k]

            for _ in range(1, num_layers):
                self.ks.append(ceil(k_step_factor*self.ks[-1]))

        self.conv_in = Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(Block(2 * hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage))

        out_dim = 2 * num_layers * hidden if readout else 2 * hidden
        self.bn = BatchNorm1d(out_dim)
        self.lin1 = Linear(out_dim, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        if isinstance(cache, list):
            self.cache = cache
        elif bool(cache):
            self.cache = self.kplex_cover.get_representations(dataset, self.ks, **cover_args)
        else:
            self.cache = None
            self.cover_args['verbose'] = False

    def reset_parameters(self):
        self.conv_in.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.bn.reset_parameters()            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def get_data(self, index):
        if self.cache is not None:
            dss = [ds[index] for ds in self.cache]
        else:
            dss = self.kplex_cover.get_representations(self.dataset[index], 
                                                       self.ks, 
                                                       **self.cover_args)

        return [self.collate(ds) for ds in dss]

    def forward(self, index):
        dss = self.get_data(index)
        data = dss[0]

        x, edge_index, weights, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = len(index)

        if self.graph_sage:
            x = F.relu(self.conv_in(x, edge_index))
        else:
            x = F.relu(self.conv_in(x, edge_index, weights))

        xs = [ 
            self.global_pool_op(x, batch, batch_size), 
            global_max_pool(x, batch, batch_size)
        ]

        for block, data in zip(self.blocks, dss[1:]):  
            cover_index, edge_index, weights, batch = data.cover_index, data.edge_index, data.edge_attr, data.batch

            x_mean = cover_pool_node(cover_index, x, data.num_nodes, pool=self.node_pool_op)
            x_max = cover_pool_node(cover_index, x, data.num_nodes, pool='max')
            x = torch.cat([x_mean, x_max], dim=1)

            if self.normalize:
                x = F.normalize(x)

            if self.graph_sage:
                x = F.relu(block(x, edge_index))
            else:
                x = F.relu(block(x, edge_index, weights))

            xs.append(self.global_pool_op(x, batch, batch_size))
            xs.append(global_max_pool(x, batch, batch_size))
        
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
        to_dense = ToDense(num_nodes=num_nodes)
        self.dataset = [to_dense(data) for data in dataset]
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

    def collate(self, data_list):
        batch = Batch()

        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])

        return batch.to(self.device)

    def forward(self, index):
        data = self.collate([self.dataset[i] for i in index.numpy().flatten()])        

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
