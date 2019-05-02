import torch
import torch_sparse
from kplex_pool import kplex_cpu, cc_cpu

def connected_components(edge_index, num_nodes=None):
    was_cuda = edge_index.is_cuda
    row, col = edge_index.cpu()

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    out = cc_cpu.connected_components(row, col, num_nodes)

    if was_cuda:
        return out.cuda()
    
    return out
