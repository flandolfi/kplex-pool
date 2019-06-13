from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.data import Batch, Data, Dataset

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify

from tqdm import tqdm


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
