from itertools import product
import sys
import argparse
import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import TUDataset

import json

import skorch
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import LRScheduler

from benchmark.model import KPlexPool
from kplex_pool.data import SkorchDataLoader, SkorchDataset

from sklearn.model_selection import GridSearchCV, StratifiedKFold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_decay_step', type=int, default=50)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--max_k', type=int, default=16)
    parser.add_argument('--max_layers', type=int, default=4)
    parser.add_argument('--k_step_factor', type=float, default=0.5)
    parser.add_argument('--to_pickle', type=str, default='results.pickle')
    parser.add_argument('--graph_sage', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    dataset = TUDataset(root='/tmp/' + args.dataset, name=args.dataset)

    net = NeuralNetClassifier(
        module=KPlexPool, 
        module__dataset=dataset,
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
        train_split=None,
        callbacks=[
            ('lr_scheduler', LRScheduler(StepLR, 
                                         step_size=args.lr_decay_step, 
                                         gamma=args.lr_decay_factor))
        ],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    net.set_params(callbacks__print_log=None)
    ks = 2**np.arange(np.floor(np.log2(args.max_k)) + 1).astype(int)

    params = {
        'module__num_layers': list(range(2, args.max_layers + 1)),
        'module__hidden': [args.hidden],
        'module__k': ks,
        'module__k_step_factor': [args.k_step_factor]
    }

    clf = GridSearchCV(estimator=net, 
                       param_grid=params, 
                       cv=StratifiedKFold(n_splits=args.folds, 
                                          shuffle=True, 
                                          random_state=42), 
                       refit=False, 
                       iid=False, 
                       scoring='accuracy', 
                       return_train_score=False,
                       n_jobs=None,
                       verbose=3)

    clf.fit(list(dataset), dataset.data.y.numpy())

    print("Best score: {}".format(clf.best_score_))
    print("Best params: {}".format(clf.best_params_))

    pd.DataFrame(clf.cv_results_).to_pickle(args.to_pickle)
