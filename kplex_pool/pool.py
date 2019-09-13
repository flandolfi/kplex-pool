import torch
import torch_sparse
import torch_scatter

from torch_geometric.utils import remove_self_loops

from kplex_pool import pool_edges_cpu



def cover_pool_node(cover_index, x, num_clusters=None, pool='mean', dense=False, mask=None):
    if dense:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        s = cover_index.unsqueeze(0) if cover_index.dim() == 2 else cover_index

        batch_size, num_nodes, feat = x.size()

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        if pool in {'add', 'mean'}:
            x = torch.matmul(s.transpose(1, 2), x)

            if pool == 'mean':
                x.div_(cover_index.sum(dim=1))
        else:
            op = getattr(torch, pool)
            x = torch.stack([op(x*(s[:, :, c].unsqueeze(-1)), dim=1)[0] for c in range(s.size(2))], dim=1)

        return x

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
        raise ValueError('Not a valid operation: %s' % pool_op)

    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1
    
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    if edge_values is None:
        edge_values = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
    
    cover_row, cover_col = cover_index.cpu()
    row, col = edge_index.cpu()
    weight = edge_values.cpu()
    
    out_row, out_col, out_weight = pool_edges_cpu.pool_edges(cover_row, cover_col, row, col, weight, pool_op, num_nodes)
    
    return torch.stack([out_row, out_col]).to(device), out_weight.to(device)


    



