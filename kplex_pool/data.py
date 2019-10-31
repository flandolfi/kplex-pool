import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch, InMemoryDataset, Dataset
from torch_geometric.transforms import ToDense



class Cover(Data):
    def __init__(self, cover_index=None, num_clusters=None, **kwargs):
        self.cover_index = cover_index

        if num_clusters is not None:
            self.__num_clusters__ = num_clusters

        super(Cover, self).__init__(**kwargs)

    def __inc__(self, key, value):
        if key == 'cover_index':
            return torch.tensor([[self.num_nodes], [self.num_clusters]])

        return super(Cover, self).__inc__(key, value)

    @property
    def num_clusters(self):
        if hasattr(self, '__num_clusters__'):
            return self.__num_clusters__
        if self.cover_index is not None:
            return self.cover_index[1].max().item() + 1
        return None

    @property
    def num_nodes(self):
        nodes = super(Cover, self).num_nodes

        if nodes is not None:
            return nodes
        if self.cover_index is not None:
            return self.cover_index[0].max().item() + 1
        return None

    @num_clusters.setter
    def num_clusters(self, num_clusters):
        self.__num_clusters__ = num_clusters

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


class DenseDataset(Dataset):
    def __init__(self, data_list):
        super(DenseDataset, self).__init__("")

        self.data = Batch()
        self.max_nodes = max([data.num_nodes for data in data_list])
        to_dense = ToDense(self.max_nodes)
        dense_list = [to_dense(data) for data in data_list]

        if 'cover_index' in data_list[0]:
            self.max_clusters = max([data.num_clusters for data in data_list])    

            for data in dense_list:
                data.cover_mask = torch.zeros(self.max_clusters, dtype=torch.uint8)
                data.cover_mask[:data.num_clusters] = 1  
                data.cover_index = torch.sparse_coo_tensor(
                        indices=data.cover_index,
                        values=torch.ones_like(data.cover_index[0]), 
                        size=torch.Size([self.max_nodes, self.max_clusters]),
                        dtype=torch.float
                    ).to_dense()

        for key in dense_list[0].keys:
            self.data[key] = default_collate([d[key] for d in dense_list])

    def __len__(self):
        if self.data.x is not None:
            return self.data.x.size(0)

        if self.data.adj is not None:
            return self.data.adj.size(0)

        return 0

    def get(self, idx):
        mask = self.data.mask[idx]
        max_nodes = mask.type(torch.uint8).argmax(-1).max().item() + 1
        out = Batch()

        for key, item in self.data('x', 'pos', 'mask'):
            out[key] = item[idx, :max_nodes]

        out.adj = self.data.adj[idx, :max_nodes, :max_nodes]
        
        if 'y' in self.data:
            out.y = self.data.y[idx]
        
        if 'cover_index' in self.data:
            cover_mask = self.data.cover_mask[idx]
            max_clusters = cover_mask.type(torch.uint8).argmax(-1).max().item() + 1
            out.cover_index = self.data.cover_index[idx, :max_nodes, :max_clusters]
            out.cover_mask = cover_mask[:, :max_clusters]

        return out

    def _download(self):
        pass

    def _process(self):
        pass
