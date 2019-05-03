import torch
import torch_sparse


def cover_pool(x, edge_index, cover_index, edge_weights=None, cover_values=None, num_nodes=None, num_clusters=None):
    if num_nodes is None:
        num_nodes = x.size(0)
    
    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1

    if edge_weights is None:
        edge_weights = torch.ones(edge_index[0].size(0), dtype=torch.float, device=edge_index.device)

    if cover_values is None:
        cover_values = torch.ones(cover_index[0].size(0), dtype=torch.float, device=cover_index.device)

    index_t, values_t = torch_sparse.transpose(cover_index, cover_values, num_nodes, num_clusters)

    out = torch_sparse.spmm(index_t, values_t, num_clusters, x)
    out_adj_index, out_adj_weights = torch_sparse.spspmm(index_t, values_t, 
        edge_index, edge_weights, num_clusters, num_nodes, num_nodes)
    out_adj_index, out_adj_weights = torch_sparse.spspmm(out_adj_index, 
        out_adj_weights, cover_index, cover_values, num_clusters, num_nodes, num_clusters)

    return out, out_adj_index, out_adj_weights
    

