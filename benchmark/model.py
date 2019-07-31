from math import ceil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch.utils.data.dataloader import default_collate
from torch_geometric.transforms import ToDense
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import (
    GCNConv, 
    SAGEConv, 
    DenseGCNConv, 
    DenseSAGEConv, 
    JumpingKnowledge, 
    TopKPooling,
    dense_diff_pool, 
    global_max_pool, 
    global_add_pool, 
    global_mean_pool
)

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify



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
    def __init__(self, dataset, hidden, k, k_step_factor=1, num_layers=2, dropout=0.3,
                 readout=True, graph_sage=False, normalize=False, simplify=False, 
                 cache_results=True, global_pool_op='add', node_pool_op='add',
                 edge_pool_op='add', num_inner_layers=2, jumping_knowledge='cat',
                 **cover_args):
        super(KPlexPool, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.simplify = simplify
        self.normalize = normalize
        self.graph_sage = graph_sage
        self.readout = readout
        self.cover_args = cover_args
        self.dataset = dataset
        self.cache_results = cache_results
        self.global_pool_op = global_add_pool if 'add' else global_mean_pool
        self.node_pool_op = node_pool_op
        self.edge_pool_op = edge_pool_op
        self.dropout = dropout

        if isinstance(k, list):
            self.ks = k
            num_layers = len(k) + 1
        else:
            self.ks = [k]

            for _ in range(1, num_layers):
                self.ks.append(ceil(k_step_factor*self.ks[-1]))

        feat = 1 if dataset.data.x is None else dataset.num_features

        self.conv_in = Block(feat, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(Block(2 * hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage))

        out_dim = 2 * num_layers * hidden if readout else 2 * hidden
        self.bn = BatchNorm1d(out_dim)
        self.lin1 = Linear(out_dim, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        if self.cache_results:
            self.preprocess_dataset()

    def preprocess_dataset(self):
        self.cache = [[(None, G) for G in self.dataset]]
        pbar = tqdm(desc="Processing dataset", total=len(self.dataset)*len(self.ks))

        for k in self.ks:
            current = []

            for _, G in self.cache[-1]:
                c_idx, clusters, _ = kplex_cover(edge_index=G.edge_index, k=k, 
                                                 num_nodes=G.num_nodes, **self.cover_args)
                edge_index, weights = cover_pool_edge(c_idx, G.edge_index, G.edge_attr, 
                                                     G.num_nodes, clusters, pool='add')

                if self.simplify:
                    edge_index, weights = simplify(edge_index, weights, num_nodes=clusters)
                
                current.append((c_idx.to(self.device), 
                                Data(edge_index=edge_index, 
                                     edge_attr=weights, 
                                     num_nodes=clusters)))
                pbar.update()
            
            self.cache.append(current)
        
        self.cache = self.cache[1:]
        pbar.close()

    def reset_parameters(self):
        self.conv_in.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.bn.reset_parameters()            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def pool_graphs(self, index, level, k, x, edge_index, weights, nodes, batch):
        if self.cache_results:
            cache = [self.cache[level][i.item()] for i in index]
            graphs = []
            cover = []
            nodes = 0
            clusters = 0
            
            for cover_index, G in cache:
                graphs.append(G)

                n, c = cover_index.clone()
                n += nodes
                c += clusters
                cover.append(torch.stack([n, c]))

                nodes = n.max().item() + 1
                clusters = c.max().item() + 1

            c_idx = torch.cat(cover, dim=1)
            data = self.collate(graphs).to(self.device)
            edge_index = data.edge_index
            weights = data.edge_attr
            batch = data.batch
        else:
            c_idx, clusters, batch = kplex_cover(edge_index=edge_index, k=k, 
                                                 num_nodes=nodes, batch=batch, **self.cover_args)
            edge_index, weights = cover_pool_edge(c_idx, edge_index, weights, nodes, clusters, 
                                                  pool=self.edge_pool_op)

            if self.simplify:
                edge_index, weights = simplify(edge_index, weights, num_nodes=clusters)

        x_mean = cover_pool_node(c_idx, x, clusters, pool=self.node_pool_op)
        x_max = cover_pool_node(c_idx, x, clusters, pool='max')
        x = torch.cat([x_mean, x_max], dim=1)

        return x, edge_index, weights, clusters, batch
    
    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def forward(self, index):
        data = self.collate([self.dataset[i.item()] for i in index])

        nodes = data.num_nodes
        edge_index = data.edge_index
        weights = data.edge_attr
        batch = data.batch
        x = data.x

        if x is None:
            x = torch.ones((nodes, 1), dtype=torch.float, device=self.device)

        batch_size = batch[-1].item() + 1

        if self.graph_sage:
            x = F.relu(self.conv_in(x, edge_index))
        else:
            x = F.relu(self.conv_in(x, edge_index, weights))

        xs = [ 
            self.global_pool_op(x, batch, batch_size), 
            global_max_pool(x, batch, batch_size)
        ]

        for level, (block, k) in enumerate(zip(self.blocks, self.ks)):    
            x, edge_index, weights, nodes, batch = self.pool_graphs(index, level, k, x, 
                                                                    edge_index, weights, 
                                                                    nodes, batch)

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

        num_nodes = ceil(ratio * dataset[0].num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        self.embed_blocks.append(Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, True))
        self.pool_blocks.append(Block(dataset.num_features, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, True))

        for _ in range(num_layers - 1):
            self.embed_blocks.append(Block(hidden, hidden, hidden, num_inner_layers, jumping_knowledge, graph_sage, True))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes, num_inner_layers, jumping_knowledge, graph_sage, True))
            num_nodes = ceil(ratio * num_nodes)

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden * 2, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
            
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def collate(self, data_list):
        batch = Batch()

        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])

        return batch

    def forward(self, index):
        data = self.collate([self.dataset[i.item()] for i in index]).to(self.device)

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


class TopK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, dropout=0.3,
                 num_inner_layers=2, jumping_knowledge='cat', graph_sage=False):
        super(TopK, self).__init__()
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
            TopKPooling(hidden, ratio) for i in range(num_layers - 1)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden * 2, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for pool in self.pools:
            pool.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def collate(self, data_list):
        return Batch.from_data_list(data_list).to(self.device)

    def forward(self, index):
        data = self.collate([self.dataset[i.item()] for i in index])
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x is None:
            x = torch.ones((data.num_nodes, 1), dtype=torch.float, device=self.device)

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
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
