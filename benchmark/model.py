from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_add_pool, global_mean_pool

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify


class KPlexPool(torch.nn.Module):
    def __init__(self, dataset, hidden, k, k_step_factor=1, num_layers=2,
                 readout=True, graph_sage=False, normalize=False, simplify=False, 
                 **cover_args):
        super(KPlexPool, self).__init__()
        self.simplify = simplify
        self.normalize = normalize
        self.graph_sage = graph_sage
        self.readout = readout
        self.cover_args = cover_args

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

    def reset_parameters(self):
        self.conv_in.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.bn.reset_parameters()            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def pool_graphs(self, x, k, edge_index, weights, nodes, batch):
        c_idx, clusters, batch = kplex_cover(edge_index=edge_index, k=k, 
                                             num_nodes=nodes, batch=batch, **self.cover_args)
        x_mean = cover_pool_node(c_idx, x, clusters, pool='add')
        x_max = cover_pool_node(c_idx, x, clusters, pool='max')
        x = torch.cat([x_mean, x_max], dim=1)
        edge_index, weights = cover_pool_edge(c_idx, edge_index, weights, nodes, clusters, pool='add')

        if self.simplify:
            edge_index, weights = simplify(edge_index, weights, num_nodes=clusters)

        return x, edge_index, weights, clusters, batch

    def forward(self, x, adj, batch):
        nodes = x.size(0)
        edge_index = adj._indices()
        weights = adj._values()

        batch_size = batch[-1].item() + 1

        if self.graph_sage:
            x = F.relu(self.conv_in(x, edge_index))
        else:
            x = F.relu(self.conv_in(x, edge_index, weights))

        xs = [ 
            global_add_pool(x, batch, batch_size), 
            global_max_pool(x, batch, batch_size)
        ]

        for block, k in zip(self.blocks, self.ks):    
            x, edge_index, weights, nodes, batch = self.pool_graphs(x, k, edge_index, weights, nodes, batch)

            if self.normalize:
                x = F.normalize(x)

            if self.graph_sage:
                x = F.relu(block(x, edge_index))
            else:
                x = F.relu(block(x, edge_index, weights))

            xs.append(global_mean_pool(x, batch, batch_size))
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
