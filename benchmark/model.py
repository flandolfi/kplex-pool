from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_add_pool, global_mean_pool

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify


class KPlexPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, k, k_step_factor=0.5, 
                 graph_sage=False, normalize=False, simplify=False):
        super(KPlexPool, self).__init__()
        self.k = k
        self.k_step_factor = k_step_factor
        self.simplify = simplify
        self.normalize = normalize
        self.graph_sage = graph_sage

        feat = 1 if dataset.data.x is None else dataset.num_features
        conv = SAGEConv if graph_sage else GCNConv

        self.conv_in = conv(feat, hidden)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(conv(2 * hidden, hidden))
        
        self.lin1 = Linear(2 * num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_in.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()
            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def pool_graphs(self, x, k, edge_index, weights, nodes, batch):
        c_idx, clusters, batch = kplex_cover(edge_index, k, nodes, batch=batch)
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

        k = self.k

        for block in self.blocks:    
            x, edge_index, weights, nodes, batch = self.pool_graphs(x, k, edge_index, weights, nodes, batch)

            if self.normalize:
                x = F.normalize(x)

            if self.graph_sage:
                x = F.relu(block(x, edge_index))
            else:
                x = F.relu(block(x, edge_index, weights))
            
            k = ceil(k*self.k_step_factor)

            xs.append(global_mean_pool(x, batch, batch_size))
            xs.append(global_max_pool(x, batch, batch_size))
        
        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
