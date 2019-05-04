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

    # --- Ugly fix ---
    # torch_sparse.spspmm produces different results if executed on GPU.
    # Move all computation on CPU.
    device = x.device
    c_idx = cover_index.cpu()
    c_val = cover_values.cpu()
    e_idx = edge_index.cpu()
    e_att = edge_weights.cpu()

    c_idx_t, c_val_t = torch_sparse.transpose(c_idx, c_val, num_nodes, num_clusters)

    out = torch_sparse.spmm(c_idx_t, c_val_t, num_clusters, x)

    out_adj_index, out_adj_weights = torch_sparse.spspmm(c_idx_t, c_val_t, e_idx, e_att, num_clusters,
                                                         num_nodes, num_nodes)
    out_adj_index, out_adj_weights = torch_sparse.spspmm(out_adj_index, out_adj_weights, c_idx, c_val,
                                                         num_clusters, num_nodes, num_clusters)

    return out.to(device), out_adj_index.to(device), out_adj_weights.to(device)
