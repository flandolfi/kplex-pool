import torch
import numpy as np
import torch_sparse

from kplex_pool.pool import cover_pool_node

from torch_geometric.utils import degree



def pool_pos(pos: dict, cover_index: torch.LongTensor, num_clusters=None):
    t_pos = torch.from_numpy(np.stack(list(pos.values()))).type(torch.float)
    t_pos = cover_pool_node(cover_index, t_pos, num_clusters)

    return dict(enumerate(t_pos.numpy()))

def pool_color(color: np.ndarray, cover_index: torch.LongTensor, num_clusters=None):
    t_color = torch.from_numpy(color.reshape((-1, 1))).type(torch.float)
    t_color = cover_pool_node(cover_index, t_color, num_clusters)

    return t_color.numpy().flatten()

def count_duplicates(cover_index: torch.LongTensor, normalize=False):
    num_nodes = cover_index[0].max().item() + 1
    duplicates = cover_index.size(1) - num_nodes

    if normalize:
        duplicates /= num_nodes
    
    return duplicates

def coverage(cover_index_list):
    last_idx = cover_index_list[0].clone()
    last_val = torch.ones_like(last_idx[0], dtype=torch.float)
    num_nodes = last_idx[0].max().item() + 1
    num_clusters = last_idx[1].max().item() + 1

    for mat in cover_index_list[1:]:
        dim = mat[1].max().item() + 1
        last_idx, last_val = torch_sparse.spspmm(last_idx, last_val,
                                                 mat, torch.ones_like(mat[0], dtype=torch.float),
                                                 num_nodes, num_clusters, dim)
        num_clusters = dim
    
    last_val = torch.ones_like(last_val)
    last_idx[0] = torch.zeros_like(last_idx[0])
    _, coverage = torch_sparse.coalesce(last_idx, last_val, 1, num_clusters)

    return coverage.numpy()/num_nodes

def node_covering_index(cover_index:torch.LongTensor, distribution=False, num_nodes=None):
    counts = torch.bincount(cover_index[0], minlength=0 if num_nodes is None else num_nodes)

    if distribution:
        counts = torch.bincount(counts)
    
    return counts

def hub_promotion(cover_index:torch.LongTensor, q=0.95, num_nodes=None, num_clusters=None, batch=None):
    counts = node_covering_index(cover_index, num_nodes=num_nodes)
    limit = np.quantile(counts.cpu().numpy(), q)
    device = cover_index.device

    if num_nodes is None:
        num_nodes = counts.size(0)
    
    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1
    
    mask = counts <= limit
    masked_index = cover_index[:, mask[cover_index[0]]]

    hub_index = (mask == 0).nonzero().view(-1)    
    out_clusters = num_clusters + hub_index.size(0)
    hub_values = torch.arange(
        start=num_clusters,
        end=out_clusters,
        device=device
    )

    out_index = torch.cat([masked_index, torch.stack([hub_index, hub_values])], dim=1)
    out_batch = None if batch is None else batch[out_index[0]] 

    return out_index, out_clusters, out_batch

def add_node_features(dataset):
    max_degree = 0.
    degrees = []
    slices = [0]

    for data in dataset:
        degrees.append(degree(data.edge_index[0], data.num_nodes, torch.float))
        max_degree = max(max_degree, degrees[-1].max().item())
        slices.append(data.num_nodes)

    dataset.data.x = torch.cat(degrees, dim=0).div_(max_degree).view(-1, 1)
    dataset.slices['x'] = torch.tensor(slices, dtype=torch.long, device=dataset.data.x.device).cumsum(0)

    return dataset
