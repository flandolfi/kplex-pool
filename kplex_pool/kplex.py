import torch

from kplex_pool import kplex_cpu
from kplex_pool.pool import cover_pool_node, cover_pool_edge
from kplex_pool.simplify import simplify
from kplex_pool.utils import hub_promotion
from kplex_pool.data import Cover, CustomDataset

from tqdm import tqdm



class KPlexCover:
    def __init__(self, cover_priority="default", kplex_priority="default", skip_covered=False):
        if cover_priority == "default":
            cover_priority = ["min_degree", "min_uncovered"]
    
        if kplex_priority == "default":
            kplex_priority = ["max_in_kplex", "max_candidates", "min_uncovered"]
    
        if not isinstance(cover_priority, list):
            cover_priority = [cover_priority]
    
        if not isinstance(kplex_priority, list):
            kplex_priority = [kplex_priority]
            
        self.cover_priority = []
        self.kplex_priority = []
        self.skip_covered = skip_covered
    
        for p in cover_priority:
            cp = getattr(kplex_cpu.NodePriority, p, None)
    
            if cp is None or cp in {
                        kplex_cpu.NodePriority.max_in_kplex,
                        kplex_cpu.NodePriority.min_in_kplex,
                        kplex_cpu.NodePriority.max_candidates,
                        kplex_cpu.NodePriority.min_candidates
                    }:
                raise ValueError('Not a valid priority: %s' % p)
            
            self.cover_priority.append(cp)
            
        for p in kplex_priority:
            kp = getattr(kplex_cpu.NodePriority, p, None)
    
            if kp is None:
                raise ValueError('Not a valid priority: %s' % p)
            
            self.kplex_priority.append(kp)
    
    def __call__(self, k, edge_index, num_nodes=None, batch=None):
        device = edge_index.device

        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1

        if batch is None:
            row, col = edge_index.cpu()
            cover_index = kplex_cpu.kplex_cover(row, col, k, int(num_nodes),
                                                self.cover_priority,
                                                self.kplex_priority,
                                                self.skip_covered).to(device)
            clusters = cover_index[1].max().item() + 1

            return cover_index, clusters, cover_index.new_zeros(clusters)

        count = batch.bincount(minlength=batch[-1] + 1)
        out_index = []
        out_batch = []
        out_clusters = 0
        min_index = 0

        for b, num_nodes in enumerate(count):
            mask = batch[edge_index[0]] == b
            cover_index, clusters, zeros = self(k, edge_index[:, mask] - min_index, num_nodes)
            cover_index[0].add_(min_index)
            cover_index[1].add_(out_clusters)

            out_index.append(cover_index)
            out_batch.append(zeros.add_(b))
            out_clusters += clusters
            min_index += num_nodes

        return torch.cat(out_index, dim=1), out_clusters, torch.cat(out_batch, dim=0)

    def process(self, dataset, k, 
                edge_pool_op='add', 
                q=None, 
                simplify=False, 
                verbose=True,
                **cover_args):
        it = tqdm(dataset, desc="Processing dataset", leave=False) if verbose else dataset
        data_list = []

        for data in it:
            cover_index, clusters, _ = self(k, data.edge_index, data.num_nodes)
            
            if q is not None:
                cover_index, clusters, _ = hub_promotion(cover_index, q=q, 
                                                         num_nodes=data.num_nodes, 
                                                         num_clusters=clusters)

            edge_index, weights = cover_pool_edge(cover_index, data.edge_index, data.edge_attr, 
                                                  data.num_nodes, clusters, pool=edge_pool_op)

            if simplify:
                edge_index, weights = simplify(edge_index, weights, num_nodes=clusters)
            
            data_list.append(Cover(cover_index=cover_index,
                                   edge_index=edge_index, 
                                   edge_attr=weights,
                                   num_covered_nodes=data.num_nodes, 
                                   num_nodes=clusters))
            
        return CustomDataset(data_list)

    def get_representations(self, dataset, ks, *args, **kwargs):
        output = [dataset]

        if (len(args) >= 4 and args[3]) or kwargs.get('verbose', True):
            ks = tqdm(ks, desc="Creating Hierarchical Representations", leave=False)

        for k in ks:
            output.append(self.process(output[-1], k, *args, **kwargs))
        
        return output

    def get_cover_fun(self, ks, dataset=None, *args, **kwargs):
        if dataset is None:
            return lambda ds, idx: self.get_representations(ds[idx], ks, *args, **kwargs)

        cache = self.get_representations(dataset, ks, *args, **kwargs)

        return lambda _, idx: [ds[idx] for ds in cache]
