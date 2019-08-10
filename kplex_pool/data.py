import torch
from torch_geometric.data import Data, InMemoryDataset

from kplex_pool import kplex_cover, cover_pool_node, cover_pool_edge, simplify
from kplex_pool.utils import hub_promotion

from tqdm import tqdm



class Cover(Data):
    def __init__(self, cover_index=None, num_covered_nodes=None, **kwargs):
        self.cover_index = cover_index

        if num_covered_nodes is not None:
            self.__num_covered_nodes__ = num_covered_nodes

        super(Cover, self).__init__(**kwargs)

    def __inc__(self, key, value):
        if key == 'cover_index':
            return torch.tensor([[self.num_covered_nodes], [self.num_nodes]])

        return super(Cover, self).__inc__(key, value)

    @property
    def num_covered_nodes(self):
        if hasattr(self, '__num_covered_nodes__'):
            return self.__num_covered_nodes__
        if self.cover_index is not None:
            return self.cover_index[0].max().item() + 1
        return None

    @property
    def num_nodes(self):
        nodes = super(Cover, self).num_nodes

        if nodes is not None:
            return nodes
        if self.cover_index is not None:
            return self.cover_index[1].max().item() + 1
        return None

    @num_covered_nodes.setter
    def num_covered_nodes(self, num_covered_nodes):
        self.__num_covered_nodes__ = num_covered_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes


class CoarsenedDataset(InMemoryDataset):
    def __init__(self, dataset, k, pool_op='add', q=None, simplify=False, verbose=True, **cover_args):
        super(CoarsenedDataset, self).__init__(dataset.root)
        
        it = tqdm(dataset, desc="Processing dataset", leave=False) if verbose else dataset
        data_list = []

        for data in it:
            cover_index, clusters, _ = kplex_cover(edge_index=data.edge_index, k=k, 
                                                   num_nodes=data.num_nodes, **cover_args)
            
            if q is not None:
                cover_index, clusters, _ = hub_promotion(cover_index, q=q, 
                                                         num_nodes=data.num_nodes, 
                                                         num_clusters=clusters)

            edge_index, weights = cover_pool_edge(cover_index, data.edge_index, data.edge_attr, 
                                                  data.num_nodes, clusters, pool=pool_op)

            if simplify:
                edge_index, weights = simplify(edge_index, weights, num_nodes=clusters)
            
            data_list.append(Cover(cover_index=cover_index,
                                   edge_index=edge_index, 
                                   edge_attr=weights,
                                   num_covered_nodes=data.num_nodes, 
                                   num_nodes=clusters))
        
        self.data, self.slices = self.collate(data_list)
        self.dataset = dataset
    
    def _download(self):
        pass

    def _process(self):
        pass
        

