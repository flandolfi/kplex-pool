import torch
import torch_sparse
import torch_scatter
from kplex_pool import pool_edges_cpu
from torch_geometric.utils import remove_self_loops


def cover_pool_node(cover_index, x, num_clusters=None, pool='mean'):
    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1
    
    xs = x.index_select(0, cover_index[0])
    pool_op = getattr(torch_scatter, "scatter_{}".format(pool))
    opts = {}

    if pool == 'min':
        opts['fill_value'] = xs.max().item() 
    elif pool == 'max':
        opts['fill_value'] = xs.min().item()

    out = pool_op(xs, cover_index[1], dim=0, dim_size=num_clusters, **opts)

    if isinstance(out, tuple):
        out = out[0]

    return out

def cover_pool_edge(cover_index, edge_index, edge_values=None, num_nodes=None, num_clusters=None, pool="add"):
    pool_op = getattr(pool_edges_cpu.PoolOp, pool, None)
    device = cover_index.device

    if pool_op is None:
        raise ValueError('Not a valid priority: %s' % pool_op)

    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1
    
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    if edge_values is None:
        edge_values = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
    
    #                              --- TO BE FIXED --- 
    # Altough the code below works on CPU, when executed on GPU does not pass all the tests.

    # if cover_index.is_cuda and (pool == 'add' or pool == 'mean'):
    #     cover_values = torch.ones(cover_index.size(1), dtype=torch.float, device=device)
    #     c_idx_t, c_val_t = torch_sparse.transpose(cover_index, cover_values, num_nodes, num_clusters)

    #     out_index, out_weights = torch_sparse.spspmm(c_idx_t, c_val_t, edge_index, edge_values, 
    #                                                  num_clusters, num_nodes, num_nodes)
    #     out_index, out_weights = torch_sparse.spspmm(out_index, out_weights, cover_index, cover_values,
    #                                                  num_clusters, num_nodes, num_clusters)

    #     if pool == 'mean':
    #         ones_idx = cover_index.new_zeros((2, num_clusters))
    #         ones_idx[1] = torch.arange(num_clusters, dtype=torch.long, device=device)
    #         ones_val = torch.ones(num_clusters, dtype=torch.float, device=device)

    #         sum_idx, sum_val = torch_sparse.spspmm(ones_idx, ones_val, out_index, out_weights, 
    #                                                1, num_clusters, num_clusters)
    #         sum_idx[0] = torch.arange(num_clusters, dtype=torch.long, device=device)
    #         sum_idx[1] = torch.arange(num_clusters, dtype=torch.long, device=device)
    #         out_index, out_weights = torch_sparse.spspmm(out_index, out_weights, sum_idx, sum_val,
    #                                                      num_clusters, num_clusters, num_clusters)
        
    #     out_index, out_weights = remove_self_loops(out_index, out_weights)
        
    #     return out_index, out_weights
    
    cover_row, cover_col = cover_index.cpu()
    row, col = edge_index.cpu()
    weight = edge_values.cpu()
    
    out_row, out_col, out_weight = pool_edges_cpu.pool_edges(cover_row, cover_col, row, col, weight, pool_op, num_nodes)
    
    return torch.stack([out_row, out_col]).to(device), out_weight.to(device)


    



