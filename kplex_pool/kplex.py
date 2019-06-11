import torch
from kplex_pool import kplex_cpu


def kplex_cover(edge_index, k, num_nodes=None, cover_priority="default", 
                kplex_priority="default", skip_covered=False, batch=None):
    if cover_priority == "default":
        cover_priority = ["min_uncovered", "min_degree"]

    if kplex_priority == "default":
        kplex_priority = ["max_in_kplex", "max_candidates", "min_uncovered", "random"]

    if not isinstance(cover_priority, list):
        cover_priority = [cover_priority]

    if not isinstance(kplex_priority, list):
        kplex_priority = [kplex_priority]
        
    cps = []
    kps = []
    device = edge_index.device

    for p in cover_priority:
        cp = getattr(kplex_cpu.NodePriority, p, None)

        if cp is None or cp in {
                    kplex_cpu.NodePriority.max_in_kplex,
                    kplex_cpu.NodePriority.min_in_kplex,
                    kplex_cpu.NodePriority.max_candidates,
                    kplex_cpu.NodePriority.min_candidates
                }:
            raise ValueError('Not a valid priority: %s' % p)
        
        cps.append(cp)
        
    for p in kplex_priority:
        kp = getattr(kplex_cpu.NodePriority, p, None)

        if kp is None:
            raise ValueError('Not a valid priority: %s' % p)
        
        kps.append(kp)

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    if batch is None:
        batch = edge_index.new_zeros(num_nodes, device=device)
    
    node_index = torch.arange(0, num_nodes, dtype=torch.long, device=device)
    batch_size = batch[-1].item() + 1
    out_clusters = 0
    out_index = []
    out_batch = []

    for b in range(batch_size):
        node_mask = batch == b
        edge_mask = node_mask.index_select(0, edge_index[0])
        batch_nodes = node_index.masked_select(node_mask)
        min_index = batch_nodes[0].item()
        r, c = edge_index - min_index
        r = r.masked_select(edge_mask).cpu()
        c = c.masked_select(edge_mask).cpu()

        index = kplex_cpu.kplex_cover(r, c, k, 
                                      batch_nodes[-1].item() - min_index + 1, 
                                      cps, kps, skip_covered)

        index[0].add_(min_index)
        clusters = index[1].max().item() + 1
        index[1].add_(out_clusters)
        
        out_index.append(index.to(device))
        out_batch.append(batch.new_ones(clusters).mul_(b))
        out_clusters += clusters

    return torch.cat(out_index, dim=1), out_clusters, torch.cat(out_batch, dim=0)



