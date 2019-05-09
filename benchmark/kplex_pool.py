from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify


class KPlexPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, k, simplify=None, keep_edges=False):
        super(KPlexPool, self).__init__()
        self.k = k
        self.simplify = simplify
        self.keep_edges = keep_edges

        self.conv_in = GCNConv(dataset.num_features, hidden)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(GCNConv(hidden, hidden))
        
        self.lin1 = Linear(2 * num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_in.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()
            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def process_input(self, x, edge_index, weights, nodes, batch):
        if self.simplify == 'pre':
            old_edges, old_weights = edge_index, weights
            edge_index, weights = simplify(edge_index, weights)

        c_idx, clusters, batch = kplex_cover(edge_index, self.k, nodes, batch=batch)
        x = cover_pool_node(c_idx, x, clusters, pool='mean')
        edge_index, weights = cover_pool_edge(c_idx, edge_index, weights, nodes, clusters, pool='add')

        if self.simplify == 'post':
            edge_index, weights = simplify(edge_index, weights)
        elif self.simplify == 'pre' and self.keep_edges:
            edge_index, weights = old_edges, old_weights

        nodes = clusters

        return x, edge_index, weights, nodes, batch

    def forward(self, data):
        x, edge_index, nodes, batch = data.x, data.edge_index, data.num_nodes, data.batch

        weights = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)

        batch_size = batch[-1].item() + 1
        x = F.relu(self.conv_in(x, edge_index, weights))
        xs = [ 
            global_mean_pool(x, batch, batch_size), 
            global_max_pool(x, batch, batch_size)
        ]

        for block in self.blocks:    
            x, edge_index, weights, nodes, batch = self.process_input(x, edge_index, weights, nodes, batch)
            x = F.relu(block(x, edge_index, weights))
            xs.append(global_mean_pool(x, batch, batch_size))
            xs.append(global_max_pool(x, batch, batch_size))
        
        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class KPlexPoolPre(KPlexPool):
    def __init__(self, k, dataset, num_layers, hidden):
        super(KPlexPoolPre, self).__init__(k, dataset, num_layers, hidden, 'pre')

class KPlexPoolPreKOE(KPlexPool):
    def __init__(self, k, dataset, num_layers, hidden):
        super(KPlexPoolPreKOE, self).__init__(k, dataset, num_layers, hidden, 'pre', True)

class KPlexPoolPost(KPlexPool):
    def __init__(self, k, dataset, num_layers, hidden):
        super(KPlexPoolPost, self).__init__(k, dataset, num_layers, hidden, 'post')
