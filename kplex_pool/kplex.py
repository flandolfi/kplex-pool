import torch
import torch_sparse
from kplex_pool import kplex_cpu


def kplex_pool(x, k, edge_index, weights=None, num_nodes=None, normalize=True,
               cover_priority="min_degree", kplex_priority="max_in_kplex"):
    row, col = edge_index
    cover = kplex_cover(row, col, k, num_nodes, normalize, cover_priority, kplex_priority)
    index, values, num_nodes, num_clusters = cover
    index_t, values_t = torch_sparse.transpose(*cover)

    if weights is None:
        weights = torch.ones(row.size(0), dtype=torch.float, device=row.device)

    out = torch_sparse.spmm(index_t, values_t, num_clusters, x)
    out_adj_index, out_adj_weights = torch_sparse.spspmm(index_t, values_t, 
        edge_index, weights, num_clusters, num_nodes, num_nodes)
    out_adj_index, out_adj_weights = torch_sparse.spspmm(out_adj_index, 
        out_adj_weights, index, values, num_clusters, num_nodes, num_clusters)
    
    return out, out_adj_index, out_adj_weights


def kplex_cover(row, col, k, num_nodes=None, normalize=True, 
                cover_priority="min_degree", kplex_priority="max_in_kplex"):
    c_priority = getattr(kplex_cpu.NodePriority, cover_priority, None)
    k_priority = getattr(kplex_cpu.NodePriority, cover_priority, None)

    if c_priority is None or c_priority is kplex_cpu.NodePriority.max_in_kplex \
            or c_priority is kplex_cpu.NodePriority.min_in_kplex:
        raise ValueError('Not a valid priority: %s' % cover_priority)
        
    if k_priority is None:
        raise ValueError('Not a valid priority: %s' % kplex_priority)

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    return kplex_cpu.kplex_cover(row, col, k, num_nodes, normalize, c_priority, k_priority)


