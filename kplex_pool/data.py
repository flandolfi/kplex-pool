import torch
from torch_geometric.data import Data, InMemoryDataset



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


class CustomDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)
    
    def _download(self):
        pass

    def _process(self):
        pass
