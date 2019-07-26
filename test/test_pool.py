import pytest
import torch
import torch_sparse
from itertools import product
from kplex_pool import kplex_cover
from kplex_pool import cover_pool_edge, cover_pool_node
from kplex_pool.kplex_cpu import NodePriority
from torch_geometric.data import Data, Batch


devices = [torch.device('cpu')]

kplex_priorities = list(NodePriority.__members__.keys())
cover_priorities = [p for p in kplex_priorities if not p.endswith("kplex") and not p.endswith("candidates")]

if torch.cuda.is_available():
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]

tests = [{
    'row': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],  # clique
    'col': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
    'k': 1,
    'cc': 1
}, {
    'row': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],  # 3-plex
    'col': [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 0, 1, 2, 0, 1],
    'k': 3,
    'cc': 1
}, {
    'row': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5], 
    'col': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4],
    'k': 1,
    'cc': 2
}]


@pytest.mark.parametrize('test,cover_priority,kplex_priority,device',
                         product(tests, cover_priorities, kplex_priorities, devices))
def test_cover_pool(test, cover_priority, kplex_priority, device):
    edge_index = torch.tensor([test['row'], test['col']], dtype=torch.long, device=device)
    k_max = test['k']
    nodes = edge_index.max().item() + 1

    for k in range(1, k_max + 1):
        index, clusters, batch = kplex_cover(edge_index, k, None, cover_priority, kplex_priority)
        x = torch.ones((nodes, 1), dtype=torch.float, device=device)

        x = cover_pool_node(index, x, clusters, pool='add')

        assert x.size(0) == clusters
        assert (x > k).sum().item() == clusters  # Every cluster contains at least k nodes
        assert batch.size(0) == clusters

        if k == k_max:
            assert x.size(0) == test['cc']
        
        index, clusters, batch = kplex_cover(edge_index, k, None, cover_priority, kplex_priority)
        x = torch.ones((nodes, 1), dtype=torch.float, device=device)

        x = cover_pool_node(index, x, clusters, pool='mean')

        assert x.size(0) == clusters
        assert x.sum().item() == clusters  # Test Normalization
        assert batch.size(0) == clusters

        if k == k_max:
            assert x.size(0) == test['cc']

        edges, weights = cover_pool_edge(index, edge_index)

        assert edges.size(1) == weights.size(0)

        if test['cc'] == 1:
            assert weights.size(0) == clusters*(clusters - 1)
        
        if k == k_max:
            assert weights.size(0) == 0


@pytest.mark.parametrize('cover_priority,kplex_priority,device',
                         product(cover_priorities, kplex_priorities, devices))
def test_cover_pool_batch(cover_priority, kplex_priority, device):
    gs = []
    ccs = 0
    k = 0
    features = 16

    for test in tests:
        edge_index = torch.tensor([test['row'], test['col']], dtype=torch.long, device=device)
        graph = Data(edge_index=edge_index)
        graph.num_nodes = edge_index.max().item() + 1
        gs.append(graph)
        ccs += test['cc']
        k = max(k, test['k'])
    
    data = Batch.from_data_list(gs).to(device)
    x = torch.ones((data.num_nodes, features), dtype=torch.float, device=device)
    index, _, batch = kplex_cover(data.edge_index, k, None, cover_priority, kplex_priority, batch=data.batch)
    x = cover_pool_node(index, x)

    assert batch.max().item() + 1 == len(tests)
    assert batch.size(0) == ccs
    assert x.size(1) == features
    assert x.size(0) == ccs
    