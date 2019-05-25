from itertools import product
import sys
import argparse
import numpy as np

import torch

from torch_geometric.datasets import TUDataset

import skorch
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit

from benchmark.model import KPlexPool
from kplex_pool.data import SkorchDataLoader, SkorchDataset, preprocess_dateset

from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--split', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--k_step_factor', type=float, default=0.5)
    parser.add_argument('--graph_sage', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    dataset = TUDataset(root='data/' + args.dataset, name=args.dataset)
    X, y = preprocess_dateset(dataset)

    net = NeuralNetClassifier(
        module=KPlexPool, 
        module__dataset=dataset,
        module__num_layers=args.layers,
        module__hidden=args.hidden,
        module__k=args.k,
        module__k_step_factor=args.k_step_factor,
        module__graph_sage=args.graph_sage,
        module__normalize=args.normalize,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=args.weight_decay,
        iterator_train=SkorchDataLoader,
        iterator_train__shuffle=True,
        iterator_valid=SkorchDataLoader,
        dataset=SkorchDataset,
        train_split=CVSplit(cv=StratifiedShuffleSplit(test_size=args.split, n_splits=1, random_state=42)),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    net.fit(X, y)
