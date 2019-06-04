import pytest
import torch
from itertools import product
from kplex_pool import kplex_cover
from kplex_pool.kplex_cpu import NodePriority


devices = [torch.device('cpu')]

kplex_priorities = list(NodePriority.__members__.keys())
cover_priorities = kplex_priorities[:-5]

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
def test_kplex_cover(test, cover_priority, kplex_priority, device):
    edge_index = torch.tensor([test['row'], test['col']], dtype=torch.long, device=device)
    k_max = test['k']
    nodes = edge_index.max().item() + 1

    for k in range(1, k_max + 1):
        index, clusters, batch = kplex_cover(edge_index, k, None, cover_priority, kplex_priority)

        if k == k_max:
            assert clusters == 1, "Parameters:\n\t" \
                                  "k = %d, k_max = %d, cover_priority = '%s', kplex_priority = '%s'.\n" \
                                  "Observed clustering:\n" \
                                  "%s" % (k, k_max, cover_priority, kplex_priority, index.__repr__())
            assert index.size(1) == nodes, "Parameters:\n\t" \
                                           "k = %d, k_max = %d, cover_priority = '%s', kplex_priority = '%s'.\n" \
                                           "Observed clustering:\n" \
                                           "%s" % (k, k_max, cover_priority, kplex_priority, index.__repr__())
        else:
            assert clusters > 1, "Parameters:\n\t" \
                                 "k = %d, k_max = %d, cover_priority = '%s', kplex_priority = '%s'.\n" \
                                 "Observed clustering:\n" \
                                 "%s" % (k, k_max, cover_priority, kplex_priority, index.__repr__())
        
        assert nodes == edge_index.max().item() + 1
        assert batch.size(0) == clusters
