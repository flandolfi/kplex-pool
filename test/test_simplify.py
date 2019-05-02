import pytest
import torch
from itertools import product
from kplex_pool.simplify import simplify
from torch_geometric.data import Data, Batch


devices = [torch.device('cpu')]

if torch.cuda.is_available():
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]

tests = [{
        'row': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'col': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        'weight': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'batch': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'min': 0,
        'max': 0,
        'len': 12
    }, {
        'row': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'col': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        'weight': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        'batch': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'min': 0,
        'max': 0,
        'len': 10
    }, {
        'row': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'col': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        'weight': [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
        'batch': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'min': 1,
        'max': 3,
        'len': 3
}]


@pytest.mark.parametrize('test,device', product(tests, devices))
def test_simplify(test, device):
    edge_index = torch.tensor([test['row'], test['col']], dtype=torch.long, device=device)
    weight =  torch.tensor(test['weight'], dtype=torch.float, device=device)
    index, weight = simplify(edge_index, weight)

    assert weight.size(0) == index.size(1)
    assert test['len'] == weight.size(0)
    assert test['min'] == weight.min()
    assert test['max'] == weight.max()

@pytest.mark.parametrize('device', devices)
def test_simplify_batch(device):
    gs = []

    for test in tests:
        edge_index = torch.tensor([test['row'], test['col']], dtype=torch.long, device=device)
        weight =  torch.tensor(test['weight'], dtype=torch.float, device=device)
        gs.append(Data(edge_index=edge_index, edge_attr=weight))
    
    batch = Batch.from_data_list(gs).to(device)
    index, weight = simplify(batch.edge_index, batch.edge_attr)

    assert weight.size(0) == index.size(1)

    for b, test in enumerate(tests):
        node_mask = batch.batch == b
        edge_mask = node_mask.index_select(0, index[0])
        batch_weight = weight.masked_select(edge_mask)

        assert test['len'] == batch_weight.size(0)
        assert test['min'] == batch_weight.min()
        assert test['max'] == batch_weight.max()

