import torch
from kplex_pool import simplify_cpu, cc_cpu


def simplify(edge_index, weights, keep_max=True, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    node_index = torch.arange(0, num_nodes, dtype=torch.long)
    device = edge_index.device
    row, col = edge_index

    sub_graphs = cc_cpu.connected_components(row.cpu(), col.cpu(), num_nodes).to(device)
    num_graphs = sub_graphs.max().item() + 1
    
    out_edges = []
    out_weights = []

    for sg in range(num_graphs):
        node_mask = sub_graphs == sg
        edge_mask = node_mask.index_select(0, row)
        sg_nodes = node_index.masked_select(node_mask)
        min_index = sg_nodes.min().item()
        r = row.masked_select(edge_mask).sub_(min_index).cpu()
        c = col.masked_select(edge_mask).sub_(min_index).cpu()
        w = weights.masked_select(edge_mask).cpu()

        r, c, w = simplify_cpu.simplify_cutoff(r, c, w, 
                                               sg_nodes.max().item() - min_index + 1,
                                               keep_max)

        out_edges.append(torch.stack([r, c], dim=0).to(device).add_(min_index))
        out_weights.append(w.to(device))
    
    return torch.cat(out_edges, dim=1), torch.cat(out_weights, dim=0)