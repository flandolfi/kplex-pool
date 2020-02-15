import torch
import torch_sparse
import torch_scatter

from torch_geometric.utils import remove_self_loops

from kplex_pool import pool_edges_cpu



def cover_pool_node(cover_index, x, num_clusters=None, pool='add', dense=False, cover_mask=None):
    """Aggregate the node features within the same k-plex, for every k-plex in
    a given cover.
    
    Args:
        cover_index (LongTensor): Cover assignment matrix, in sparse 
            coordinate form. It can assign nodes of different graphs in a
            batch.
        x (FloatTensor): Feature matrix of the nodes in the graph(s).
        num_clusters (int, optional): Number of total k-plexes. Defaults to 
            `None`.
        pool (str, optional): Aggregation function (`"add"`, `"mean"`, `"min"`
            or `"max"`). Defaults to `"add"`.
        dense (bool, optional): If `True`, compute the aggregation in dense
            graph form. Defaults to `False`.
        cover_mask (ByteTensor, optional): Boolean tensor representing the
            columns of the cover assignment matrix that contain significant
            data. Can be used only if `dense` is `True`. Defaults to `None`.
    
    Returns:
        FloatTensor: The feature matrix of the coarsened graph.
    """
    if dense:
        out = x.unsqueeze(0) if x.dim() == 2 else x
        s = cover_index.unsqueeze(0) if cover_index.dim() == 2 else cover_index
        batch_size, _, clusters = s.size()

        if pool in {'add', 'mean'}:
            out = torch.bmm(s.transpose(1, 2), out)

            if pool == 'mean':
                out = out / s.sum(dim=1).unsqueeze(-1).clamp(min=1)
        else:
            op = getattr(torch, pool if pool is not "add" else "sum")
            out = op(out.unsqueeze(2).repeat(1, 1, clusters, 1) * s.unsqueeze(-1), dim=1)

            if isinstance(out, tuple):
                out = out[0]
        
        if cover_mask is not None:
            out = out * cover_mask.view(batch_size, clusters, 1).to(x.dtype)

        return out

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
    """For every two k-plexes in a given cover, aggregate the weights of all
    the edges having its endvertices on both of them.
    
    Args:
        cover_index (LongTensor): Cover assignment matrix, in sparse 
            coordinate form. It can assign nodes of different graphs in a
            batch.
        edge_index (LongTensor): Edge coordinate matrix.
        edge_values (FloatTensor, optional): Weights of the edges. If `None`,
            defaults to a vector of ones. Defaults to `None`.
        num_nodes (int, optional): Number of total nodes. Defaults to None.
        num_clusters (int, optional): Number of total k-plexes. Defaults to 
            `None`.
        pool (str, optional): Edge agregation function (`"add"`, `"mul"`, 
            `"mean"`, `"min"` or `"max"`). Defaults to "add".
    
    Raises:
        ValueError: If provided an undefined aggregation function.
    
    Returns:
        (LongTensor, FloatTensor): Sparse coordinate representation of the
            coarsened graphs.
    """
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


    



