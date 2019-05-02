import torch
from kplex_pool import simplify_cpu, cc_cpu


def simplify(edge_index, weights, keep_max=True, num_nodes=None):
    was_cuda = edge_index.is_cuda or weights.is_cuda
    row, col = edge_index.cpu()
    weights = weights.cpu()

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    sub_graphs = cc_cpu.connected_components(row, col, num_nodes)
    num_graphs = sub_graphs.max().item() + 1
    
    out_edges = []
    out_weights = []

    for b in range(num_graphs):
        node_mask = sub_graphs == b
        edge_mask = node_mask.index_select(0, row)
        r = row.masked_select(edge_mask)
        c = col.masked_select(edge_mask)
        w = weights.masked_select(edge_mask)
        min_index = min(r.min().item(), c.min().item())
        batch_nodes = max(r.max().item(), c.max().item()) + 1 - min_index
        r = r.add(-min_index)
        c = c.add(-min_index)

        r, c, w = simplify_cpu.simplify_cutoff(r, c, w, batch_nodes, keep_max)

        out_edges.append(torch.stack([r, c], dim=0).add(min_index))
        out_weights.append(w)
    
    out_edges = torch.cat(out_edges, dim=1)
    out_weights = torch.cat(out_weights, dim=0)

    if was_cuda:
        return out_edges.cuda(), out_weights.cuda()
    
    return out_edges, out_weights