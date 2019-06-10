import torch
import numpy as np
import torch_sparse

from kplex_pool import cover_pool_node



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
    last_idx = cover_index_list[0]
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
