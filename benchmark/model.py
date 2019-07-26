from math import ceil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch.utils.data.dataloader import default_collate
from torch_geometric.transforms import ToDense
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import (
    GraphConv, 
    TopKPooling,
    DenseSAGEConv, 
    dense_diff_pool, 
    JumpingKnowledge, 
    GCNConv, SAGEConv, 
    global_max_pool, 
    global_add_pool, 
    global_mean_pool
)

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify


class KPlexPool(torch.nn.Module):
    def __init__(self, dataset, hidden, k, k_step_factor=1, num_layers=2,
                 readout=True, graph_sage=False, normalize=False, simplify=False, 
                 cache_results=True, global_pool_op='add', node_pool_op='add',
                 edge_pool_op='add', **cover_args):
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

        if isinstance(k, list):
            self.ks = k
            num_layers = len(k) + 1
        else:
            self.ks = [k]

            for _ in range(1, num_layers):
                self.ks.append(ceil(k_step_factor*self.ks[-1]))

        feat = 1 if dataset.data.x is None else dataset.num_features
        conv = SAGEConv if graph_sage else GCNConv

        self.conv_in = conv(feat, hidden)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(conv(2 * hidden, hidden))

        out_dim = 2 * num_layers * hidden if readout else 2 * hidden
        self.bn = BatchNorm1d(out_dim)
        self.lin1 = Linear(out_dim, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        if self.cache_results:
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
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

# Other models used for comparison, slightly modified from
# https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/kernel

class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat'):
        super(Block, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_channels + out_channels, out_channels)
        else:
            self.lin = Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        return self.lin(self.jump([x1, x2]))


class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.25):
        super(DiffPool, self).__init__()
        num_nodes = max([data.num_nodes for data in dataset])
        to_dense = ToDense(num_nodes=num_nodes)
        self.dataset = [to_dense(data) for data in dataset]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()

        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
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

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x.mean(dim=1)]
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, l, e = dense_diff_pool(x, adj, s)
                link_loss += l
                ent_loss += e

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1), link_loss, ent_loss

    def __repr__(self):
        return self.__class__.__name__


class TopK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8):
        super(TopK, self).__init__()
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [TopKPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
