import torch
import skorch
from torch_geometric.data import Batch


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
        x = data.x if data.x is not None else torch.ones((data.num_nodes, 1), 
                                                         dtype=torch.float, 
                                                         device=data.edge_index.device)
        edge_attr = data.edge_attr if data.edge_attr else torch.ones_like(data.edge_index[0], 
                                                                          dtype=torch.float)  
        
        return {
            'x': x,
            'adj': torch.sparse.FloatTensor(data.edge_index, 
                                            edge_attr, 
                                            size=[data.num_nodes, data.num_nodes], 
                                            device=x.device),
            'batch': data.batch
        }, data.y
    
        
class SkorchDataset(skorch.dataset.Dataset):
    def __init__(self, X, y):        
        super(SkorchDataset, self).__init__(X, y=y, length=len(X))
    
    def transform(self, X, y):
        return X
