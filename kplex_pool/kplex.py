import torch

from kplex_pool import kplex_cpu
from kplex_pool.pool import cover_pool_node, cover_pool_edge
from kplex_pool.simplify import simplify as simplify_graph
from kplex_pool.utils import hub_promotion
from kplex_pool.data import Cover, CustomDataset, DenseDataset

from tqdm import tqdm



class KPlexCover:
    """KPlexCover Algorithm.
    
    Args:
        cover_priority (str or list, optional): Priority used to extract the 
            pivot node (`"random"`, `"min_degree"`, `"max_degree"`, 
            `"min_uncovered"`, `"max_uncovered"`, `"min_in_kplex"`, 
            `"max_in_kplex"`, `"min_candidates"`, `"max_candidates"`, or 
            `"default"`). Defaults to `"default"`.
        kplex_priority (str or list, optional): Priority used to extract the
            next k-plex candidate (`"random"`, `"min_degree"`, `"max_degree"`, 
            `"min_uncovered"`, `"max_uncovered"`, `"min_in_kplex"`, 
            `"max_in_kplex"`, `"min_candidates"`, `"max_candidates"`, or 
            `"default"`). Defaults to `"default"`.
        skip_covered (bool, optional): Give max priority to uncovered nodes.
            Defaults to `False`.
    
    Raises:
        ValueError: A given priority is not defined.
    """

    def __init__(self, cover_priority="default", kplex_priority="default", skip_covered=False):
        if cover_priority == "default":
            cover_priority = ["min_degree", "min_uncovered"]
    
        if kplex_priority == "default":
            kplex_priority = ["max_in_kplex", "max_candidates", "max_uncovered"]
    
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
        """Compute the k-plex cover of a given graph or batch of graphs.
        
        Args:
            k (int): Number of maximum missing links per node. Must be at
                least 1.
            edge_index (LongTensor): Edge coordinates (sparse COO matrix 
                form).
            num_nodes (int, optional): Number of (total) nodes. Defaults to
                `None`.
            batch (LongTensor, optional): Batch vector, assigning every node
                to a specific example in the batch. Defaults to `None`.
        
        Returns:
            (LongTensor, int, LongTensor): A cover index matrix, assigning
                every node to a specific k-plex in the cover; the number of
                k-plexes; a batch vector assigning every k-plex to a specific
                example in the batch.
        """
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
                verbose=True):
        """Compute the k-plex cover for a whole dataset of graphs and 
        post-process it.
        
        Args:
            dataset (torch_geometric.Dataset): A graph dataset.
            k (int): Number of maximum missing links per node. Must be at
                least 1.
            edge_pool_op (str, optional): Edge-weights aggregation funciton 
                (`"add"`, `"mul"`,` "max"`, `"min"`, or `"mean"`). Defaults
                to `"add"`.
            q (float, optional): Hub-promotion quantile threshold (must be a
                float in [0, 1]). Defaults to `None`.
            simplify (bool, optional): Apply simplification to coarsened
                grpahs. Defaults to `False`.
            verbose (bool, optional): Show a progress bar. Defaults to `True`.
        
        Returns:
            (CustomDataset, CustomDataset): The input dataset, augmented with
                `"cover_index"` and `"num_clusters"` keys, and the coarsened
                dataset (with no node features).
        """
        it = tqdm(dataset, desc="Processing dataset", leave=False) if verbose else dataset
        in_list = []
        out_list = []
        
        for data in it:
            cover_index, clusters, _ = self(k, data.edge_index, data.num_nodes)
            
            if q is not None:
                cover_index, clusters, _ = hub_promotion(cover_index, q=q, 
                                                         num_nodes=data.num_nodes, 
                                                         num_clusters=clusters)

            edge_index, weights = cover_pool_edge(cover_index, data.edge_index, data.edge_attr, 
                                                  data.num_nodes, clusters, pool=edge_pool_op)

            if simplify:
                edge_index, weights = simplify_graph(edge_index, weights, num_nodes=clusters)
            
            keys = dict(data.__iter__())
            keys['num_nodes'] = data.num_nodes
            in_list.append(Cover(cover_index=cover_index, num_clusters=clusters, **keys))
            out_list.append(Cover(edge_index=edge_index, edge_attr=weights, num_nodes=clusters))
        
        return CustomDataset(in_list), CustomDataset(out_list)

    def get_representations(self, dataset, ks, verbose=True, *args, **kwargs):
        """Build a hierarchy of graphs for each graph in a given dataset.
        
        Args:
            dataset (torch_geometric.Dataset): A graph dataset.
            ks (list): A list of k parameters, one for each layer of the 
                hierarchy.
            verbose (bool, optional): Show a progress bar. Defaults to True.
        
        Returns:
            list: A list of `CustomDataset`s, where every dataset (apart from 
                the first one) contains at a given index the coarsened verison
                of the graph at the same index in the previous dataset in the 
                list. 
        """
        last_dataset = dataset
        output = []

        if verbose:
            ks = tqdm(ks, desc="Creating Hierarchical Representations", leave=False)

        for k in ks:
            cover, last_dataset = self.process(last_dataset, k, *args, **kwargs)
            output.append(cover)

        output.append(last_dataset)
        
        return output

    def get_cover_fun(self, ks, dataset=None, dense=False, *args, **kwargs):
        """Build and return a function that, for a given dataset and a set of
        indices, computes and returns the graph hierarchies at that indices. 
        If `dataset` is not `None`, the hiearachies are precomputed for that 
        dataset and the returned function will ignore the first parameter. 
        
        Args:
            ks (list): A list of k parameters, one for each layer of the 
                hierarchy.
            dataset (torch_geometric.Dataset, optional): A graph dataset. 
                Defaults to None.
            dense (int or bool, optional): The datasets returned by the 
                function will be dense starting from the given layer. `True`
                acts as 0, while `False` as `len(ks) + 1`. Defaults to
                `False`.
        
        Returns:
            callable: The graph-hierarchy function.
        """
        dense = int(not dense)*(len(ks) + 1) if isinstance(dense, bool) else dense

        def cover_fun(ds, idx):
            hierarchy = self.get_representations(ds[idx], ks, *args, **kwargs)
            
            return [DenseDataset(ds) if l >= dense else ds for l, ds in enumerate(hierarchy)]

        if dataset is None:
            return cover_fun

        cache = cover_fun(dataset, slice(None))

        return lambda _, idx: [ds[idx] for ds in cache]
