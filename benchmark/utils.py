import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch_geometric.data import Data

import cudf
import cugraph as cx


def to_cugraph(data: Data):
    df = data.edge_index.contiguous().t().clone().detach()
    df = cudf.from_dlpack(to_dlpack(df))

    if data.edge_attr is not None:
        weights = data.edge_attr.contiguous().view(-1).clone().detach()
        df[2] = cudf.from_dlpack(to_dlpack(weights))

    return cx.from_cudf_edgelist(df, source=0, destination=1,
                                 edge_attr=2 if df.shape[1] == 3 else None,
                                 renumber=False)


def from_cudf(df: cudf.DataFrame):
    return from_dlpack(df.to_dlpack())
