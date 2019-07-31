from itertools import product
import sys
import argparse
import numpy as np

import torch

from torch_geometric.datasets import TUDataset

import skorch
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit

from benchmark import model

from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KPlexPool')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--cover_priority', type=str, default='default')
    parser.add_argument('--kplex_priority', type=str, default='default')
    parser.add_argument('--global_pool_op', type=str, default='add')
    parser.add_argument('--node_pool_op', type=str, default='add')
    parser.add_argument('--edge_pool_op', type=str, default='add')
    parser.add_argument('--jumping_knowledge', type=str, default='cat')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--split', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--inner_layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--k_step_factor', type=float, default=0.5)
    parser.add_argument('--graph_sage', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--skip_covered', action='store_true')
    parser.add_argument('--no_readout', action='store_false')
    parser.add_argument('--no_cache', action='store_false')
    parser.add_argument('--ks', nargs='*', type=int)
    args = parser.parse_args()

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    dataset = TUDataset(root='data/' + args.dataset, name=args.dataset)
    X = np.arange(len(dataset)).reshape((-1, 1))
    y = dataset.data.y.numpy()

    params = {
        'module': getattr(model, args.model), 
        'module__dataset': dataset,
        'module__num_layers': args.layers,
        'module__hidden': args.hidden,
        'module__graph_sage': args.graph_sage,
        'module__dropout': args.dropout,
        'module__num_inner_layers': args.inner_layers,
        'module__jumping_knowledge': args.jumping_knowledge,
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'criterion': model.PoolLoss if args.model == 'DiffPool' else torch.nn.modules.loss.NLLLoss,
        'optimizer': torch.optim.Adam,
        'optimizer__weight_decay': args.weight_decay,
        'iterator_train__shuffle': True,
        'train_split': CVSplit(cv=StratifiedShuffleSplit(test_size=args.split, n_splits=1, random_state=42)),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    if args.model == 'KPlexPool':
        params.update(
            module__k=args.k if args.ks is None else args.ks,
            module__k_step_factor=args.k_step_factor,
            module__normalize=args.normalize,
            module__readout=args.no_readout,
            module__skip_covered=args.skip_covered,
            module__cache_results=args.no_cache,
            module__cover_priority=args.cover_priority,
            module__kplex_priority=args.kplex_priority,
            module__global_pool_op=args.global_pool_op,
            module__node_pool_op=args.node_pool_op,
            module__edge_pool_op=args.edge_pool_op
        )
    else:
        params.update(module__ratio=args.ratio)

    NeuralNetClassifier(**params).fit(X, y)
