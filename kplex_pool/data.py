import torch
import skorch
from torch_geometric.data import Batch, Data, Dataset
from torch_sparse import coalesce


class SkorchDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(SkorchDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: self._collate_fn(data_list, follow_batch),
            **kwargs)
    
    def _collate_fn(self, data_list, follow_batch=[]):
        data = Batch.from_data_list(data_list, follow_batch)
        
        return {
            'x': data.x,
            'adj': torch.sparse.FloatTensor(data.edge_index, 
                                            data.edge_attr, 
                                            size=[data.num_nodes, data.num_nodes], 
                                            device=data.x.device),
            'batch': data.batch
        }, data.y
    
        
class SkorchDataset(skorch.dataset.Dataset):
    def __init__(self, X, y):        
        super(SkorchDataset, self).__init__(X, y=y, length=len(X))
    
    def transform(self, X, y):
        return X

def preprocess_dateset(dataset: Dataset, op='add', fill_value=0):
    X = []
    y = dataset.data.y.numpy()

    for data in dataset:
        x = data.x if data.x is not None else torch.ones((data.num_nodes, 1), 
                                                         dtype=torch.float, 
                                                         device=data.edge_index.device)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if data.edge_attr is not None else torch.ones_like(edge_index[0], 
                                                                                 dtype=torch.float)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, 
                                         edge_index.size(0), edge_index.size(1), 
                                         op, fill_value)
        X.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y))
    
    return X, y