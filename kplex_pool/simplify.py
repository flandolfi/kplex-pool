import torch
from kplex_pool import simplify_cpu


def simplify(edge_index, weights, num_nodes=None, fair=True):
    row, col = edge_index

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    row, col, weights = simplify_cpu.remove_lightest(row, col, weights, num_nodes, fair)
    
    return torch.stack([row, col], dim=0), weights