import pytest
import torch
import torch_sparse
from itertools import product
from kplex_pool import kplex_cover
from kplex_pool import cover_pool
from kplex_pool.kplex_cpu import NodePriority


devices = [torch.device('cpu')]

kplex_priorities = list(NodePriority.__members__.keys())
cover_priorities = kplex_priorities[:-2]

if torch.cuda.is_available():
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]

tests = [{
        'row': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],  # clique
        'col': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        'k': 1
    }, {
        'row': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],  # 3-plex
        'col': [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 0, 1, 2, 0, 1],
        'k': 3
}]


@pytest.mark.parametrize('test,cover_priority,kplex_priority,device',
                         product(tests, cover_priorities, kplex_priorities, devices))
def test_cover_pool(test, cover_priority, kplex_priority, device):
    edge_index = torch.tensor([test['row'], test['col']], dtype=torch.long, device=device)
    k_max = test['k']

    for k in range(1, k_max + 1):
        index, values, nodes, clusters = kplex_cover(edge_index, k, None, False, cover_priority, kplex_priority)
        x = torch.ones((nodes, 1), dtype=torch.float, device=device)

        x, idx, w = cover_pool(x, edge_index, index, cover_values=values)

        assert x.size(0) == clusters
        assert (x > k).sum().item() == clusters  # Every cluster contains at least k nodes

        index, values, nodes, clusters = kplex_cover(edge_index, k, None, True, cover_priority, kplex_priority)
        x = torch.ones((nodes, 1), dtype=torch.float, device=device)

        x, idx, w = cover_pool(x, edge_index, index, cover_values=values)

        assert x.size(0) == clusters
        assert x.sum().item() == nodes  # Test Normalization
