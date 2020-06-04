import numpy as np
from os import path

import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch, InMemoryDataset, Dataset, download_url
from torch_geometric.transforms import ToDense



class Cover(Data):
    """Augment a Data object with `"cover_index"` and `num_clusters"` keys. 
    Support slicing on cover index matrices.

    Args:
        cover_index (LongTensor, optional): A cover index matrix (in sparse
            coordinate form), assigning every node to a specific k-plex in the
            cover. Defaults to None.
        num_clusters (int, optional): Number of k-plexes in the cover matrix. 
            Defaults to None.
    """
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
    """Create a dataset from a `torch_geometric.Data` list.
    
    Args:
        data_list (list): List of graphs.
    """
    def __init__(self, data_list):
        super(CustomDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)
    
    def _download(self):
        pass

    def _process(self):
        pass


class DenseDataset(Dataset):
    """Dense Graphs Dataset.
    
    Args:
        data_list (list): list of graphs.
    """
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

    def len(self):
        if self.data.x is not None:
            return self.data.x.size(0)

        if 'adj' in self.data:
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

    def index_select(self, idx):
        return self.get(idx)

    def _download(self):
        pass

    def _process(self):
        pass


class NDPDataset(InMemoryDataset):
    """The synthetic dataset from `"Hierarchical Representation Learning in 
    Graph Neural Networks with Node Decimation Pooling"
    <https://arxiv.org/abs/1910.11436>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If `"train"`, loads the training dataset.
            If `"val"`, loads the validation dataset.
            If `"test"`, loads the test dataset. Defaults to `"train"`.
        easy (bool, optional): If `True`, use the easy version of the dataset.
            Defaults to `True`.
        small (bool, optional): If `True`, use the small version of the
            dataset. Defaults to `True`.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to `None`.
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. Defaults to `None`.
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. Defaults to `None`.
    """
    base_url = ('http://github.com/FilippoMB/'
                'Benchmark_dataset_for_graph_classification/'
                'raw/master/datasets/')
    
    def __init__(self, root, split='train', easy=True, small=True, transform=None, pre_transform=None, pre_filter=None):
        self.file_name = ('easy' if easy else 'hard') + ('_small' if small else '')
        self.split = split.lower()

        assert self.split in {'train', 'val', 'test'}

        if self.split != 'val':
            self.split = self.split[:2]
        
        super(NDPDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return '{}.npz'.format(self.file_name)
    
    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.file_name)

    def download(self):
        download_url('{}{}.npz'.format(self.base_url, self.file_name), self.raw_dir)

    def process(self):
        npz = np.load(path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        raw_data = (npz['{}_{}'.format(self.split, key)] for key in ['feat', 'adj', 'class']) 
        data_list = [Data(x=torch.FloatTensor(x), 
                          edge_index=torch.LongTensor(np.stack(adj.nonzero())), 
                          y=torch.LongTensor(y.nonzero()[0])) for x, adj, y in zip(*raw_data)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
