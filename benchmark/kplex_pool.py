from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool

from kplex_pool import kplex_cover, cover_pool, simplify


class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                   out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))

        return self.lin(torch.cat([x1, x2], dim=-1))


class KPlexPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, k, simplify=None):
        super(KPlexPool, self).__init__()
        self.k = k
        self.simplify = simplify

        self.in_block = Block(dataset.num_features, hidden, hidden)
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.blocks.append(Block(hidden, hidden, hidden))

        self.lin1 = Linear(num_layers*hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.in_block.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, weights, nodes, batch = data.x, data.edge_index, data.edge_attr, data.num_nodes, data.batch

        x = F.relu(self.in_block(x, edge_index))
        xs = [global_mean_pool(x, batch)]

        for embed in self.blocks:            
            if self.simplify == 'pre':
                edge_index, weights = simplify(edge_index, weights)

            c_idx, c_val, nodes, clusters = kplex_cover(edge_index, self.k, nodes)
            x, edge_index, weights, batch = cover_pool(x, edge_index, c_idx, weights, c_val, nodes, clusters, batch)
            nodes = clusters

            if self.simplify == 'post':
                edge_index, weights = simplify(edge_index, weights)

            x = F.relu(embed(x, edge_index))
            xs.append(global_mean_pool(x, batch))

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


class KPlexPoolPost(KPlexPool):
    def __init__(self, k, dataset, num_layers, hidden):
        super(KPlexPoolPost, self).__init__(k, dataset, num_layers, hidden, 'post')
