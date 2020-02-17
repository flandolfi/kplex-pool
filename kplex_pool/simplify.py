import torch
from kplex_pool import simplify_cpu, cc_cpu


def simplify(edge_index, edge_attr, keep_max=True, num_nodes=None):
    """Sparsify the input graph by removing every edge with weight lower (or
    higher) than the highest (lowest) threshold value such that the resulting
    graph has the same components of the input one.
    
    Args:
        edge_index (LongTensor): Edge coordinate matrix.
        edge_attr (FloatTensor): Weights of the edges. 
        keep_max (bool, optional): If `True`, the algorithm drops the weights
            with the lowest values and finds the highest threshold value. 
            Viceversa if `False`.
        num_nodes (int, optional): Number of total nodes in the graph. 
            Defaults to `None`.
    
    Returns:
        (LongTensor, FloatTensor): The simplified graph, in sparse coordinate
            form.
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    device = edge_index.device
    node_index = torch.arange(0, num_nodes, dtype=torch.long, device='cpu')
    row, col = edge_index.cpu()

    sub_graphs = cc_cpu.connected_components(row, col, num_nodes)
    num_graphs = sub_graphs.max().item() + 1
    weights = edge_attr.cpu()
    
    out_edges = []
    out_weights = []

    for sg in range(num_graphs):
        node_mask = sub_graphs == sg
        edge_mask = node_mask.index_select(0, row)
        sg_nodes = node_index.masked_select(node_mask)
        min_index = sg_nodes.min().item()
        r = row.masked_select(edge_mask).sub_(min_index)
        c = col.masked_select(edge_mask).sub_(min_index)
        w = weights.masked_select(edge_mask)

        r, c, w = simplify_cpu.simplify_cutoff(r, c, w, 
                                               sg_nodes.max().item() - min_index + 1,
                                               keep_max)

        out_edges.append(torch.stack([r, c], dim=0).to(device).add_(min_index))
        out_weights.append(w.to(device))
    
    return torch.cat(out_edges, dim=1), torch.cat(out_weights, dim=0)
