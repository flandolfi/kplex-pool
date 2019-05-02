import torch
import torch_sparse
from kplex_pool import kplex_cpu, cc_cpu

def connected_components(edge_index, num_nodes=None):
    row, col = edge_index

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    return cc_cpu.connected_components(row, col, num_nodes)


