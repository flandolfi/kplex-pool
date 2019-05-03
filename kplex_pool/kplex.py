import torch
from kplex_pool import kplex_cpu


def kplex_cover(edge_index, k, num_nodes=None, normalize=True, 
                cover_priority="min_degree", kplex_priority="max_in_kplex", batch=None):
    c_priority = getattr(kplex_cpu.NodePriority, cover_priority, None)
    k_priority = getattr(kplex_cpu.NodePriority, kplex_priority, None)

    if c_priority is None or c_priority is kplex_cpu.NodePriority.max_in_kplex \
            or c_priority is kplex_cpu.NodePriority.min_in_kplex:
        raise ValueError('Not a valid priority: %s' % cover_priority)
        
    if k_priority is None:
        raise ValueError('Not a valid priority: %s' % kplex_priority)

    device = edge_index.device
    row, col = edge_index.cpu()

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    if batch is None:
        batch = edge_index.new_zeros(num_nodes)

    index, values, nodes, clusters, batch = kplex_cpu.kplex_cover(row, col, k, num_nodes, normalize, c_priority, k_priority, batch)

    return index.to(device), values.to(device), nodes, clusters, batch.to(device)



