import torch
import torch_sparse
from kplex_pool import kplex_cpu


def kplex_pool(edge_index, k, num_nodes=None, cover_priority="min_degree", kplex_priority="max_in_kplex"):
    row, col = edge_index
    index, values, num_nodes, num_clusters = kplex_cover(row, col, k, num_nodes, cover_priority, kplex_priority)

    # TODO


def kplex_cover(row, col, k, num_nodes=None, cover_priority="min_degree", kplex_priority="max_in_kplex"):
    c_priority = getattr(kplex_cpu.NodePriority, cover_priority, None)
    k_priority = getattr(kplex_cpu.NodePriority, cover_priority, None)

    if c_priority is None or c_priority is kplex_cpu.NodePriority.max_in_kplex \
            or c_priority is kplex_cpu.NodePriority.min_in_kplex:
        raise ValueError('Not a valid priority: %s' % cover_priority)
        
    if k_priority is None:
        raise ValueError('Not a valid priority: %s' % kplex_priority)

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    return kplex_cpu.kplex_cover(row, col, k, num_nodes, c_priority, k_priority)


