import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

import cudf
import cugraph as cx


def to_cugraph(data: Data):
    edge_index, weights = add_self_loops(data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
    df = edge_index.contiguous().t().clone().detach()
    df = cudf.from_dlpack(to_dlpack(df))

    if weights is not None:
        weights = weights.contiguous().view(-1).clone().detach()
        df[2] = cudf.from_dlpack(to_dlpack(weights))

    return cx.from_cudf_edgelist(df, source=0, destination=1,
                                 edge_attr=2 if df.shape[1] == 3 else None,
                                 renumber=False)


def from_cudf(df: cudf.DataFrame):
    return from_dlpack(df.to_dlpack())
