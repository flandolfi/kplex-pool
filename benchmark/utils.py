import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch_geometric.data import Data

import cudf
import cugraph as cx


def to_cugraph(graph: Data):
    data = graph.edge_index.T.clone().detach()

    if graph.edge_attr is not None:
        data = torch.cat((data, graph.edge_attr.view(-1, 1).clone().detach()), dim=-1)

    df = cudf.from_dlpack(to_dlpack(data))
    return cx.from_cudf_edgelist(df, source='0', destination='1',
                                 edge_attr='2' if graph.edge_attr else None,
                                 renumber=False)


def from_cudf(df: cudf.DataFrame):
    return from_dlpack(df.to_dlpack())
