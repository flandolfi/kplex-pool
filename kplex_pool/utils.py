import torch
import numpy as np
from kplex_pool import cover_pool_node


def pool_pos(pos: dict, cover_index: torch.LongTensor, num_clusters=None):
    t_pos = torch.from_numpy(np.stack(list(pos.values()))).type(torch.float)
    t_pos = cover_pool_node(cover_index, t_pos, num_clusters)

    return dict(enumerate(t_pos.numpy()))

def pool_color(color: np.ndarray, cover_index: torch.LongTensor, num_clusters=None):
    t_color = torch.from_numpy(color.reshape((-1, 1))).type(torch.float)
    t_color = cover_pool_node(cover_index, t_color, num_clusters)

    return t_color.numpy().flatten()
