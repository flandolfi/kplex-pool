import torch
import torch_sparse
import torch_scatter


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
    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1
    
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    out_row = []
    out_col = []
    out_w = []
    row, col = edge_index

    if edge_values is None:
        edge_values = torch.ones_like(row, dtype=torch.float)

    for k_row in range(num_clusters):
        k_row_nodes = cover_index.masked_select(cover_index[1] == k_row)
        node_mask_row = edge_index.new_zeros(num_nodes).byte()
        node_mask_row[k_row_nodes] = True
        edge_mask_row = node_mask_row.index_select(0, row)
        edge_mask_col = node_mask_row.index_select(0, col)

        self_loops = edge_values.masked_select(edge_mask_col * edge_mask_row)
        out_w.append(self_loops)
        out_col.append(k_row * torch.ones_like(self_loops, dtype=torch.long))
        out_row.append(k_row * torch.ones_like(self_loops, dtype=torch.long))

        for k_col in range(num_clusters):
            if k_col == k_row:
                continue
            
            k_col_nodes = cover_index.masked_select(cover_index[1] == k_col)
            node_mask_col = edge_index.new_zeros(num_nodes).byte()
            node_mask_col[k_col_nodes] = True
            edge_mask_col = node_mask_col.index_select(0, col)

            edges = edge_values.masked_select(edge_mask_col * edge_mask_row)
            out_w.append(edges)
            out_col.append(k_row * torch.ones_like(edges, dtype=torch.long))
            out_row.append(k_row * torch.ones_like(edges, dtype=torch.long))
            
    out_row = torch.cat(out_row)
    out_col = torch.cat(out_col)
    out_w = torch.cat(out_w)

    out_index = torch.stack([out_row, out_col])
    fill_value = 0

    if pool == 'min':
        fill_value = out_w.max().item() 
    elif pool == 'max':
        fill_value = out_w.min().item()
    elif pool == 'mul' or pool == 'div':
        fill_value = 1

    return torch_sparse.coalesce(out_index, out_w, num_clusters, num_clusters, 
                                 op=pool, fill_value=fill_value)



    



