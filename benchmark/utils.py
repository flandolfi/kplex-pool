import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch_geometric.data import Data

import cudf
import cugraph as cx


def to_cugraph(data: Data):
    df = data.edge_index

    if data.edge_attr is not None:
        df = torch.cat((df, data.edge_attr.view(1, -1)), dim=0)

    df = df.clone().detach().contiguous().t()
    df = cudf.from_dlpack(to_dlpack(df))

    return cx.from_cudf_edgelist(df, source=0, destination=1,
                                 edge_attr=None if data.edge_attr is None else 2,
                                 renumber=False)


def from_cudf(df: cudf.DataFrame):
    return from_dlpack(df.to_dlpack())
