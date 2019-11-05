import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product

import torch

from torch_geometric.datasets import TUDataset

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from benchmark import model
from kplex_pool import KPlexCover
from kplex_pool.utils import add_node_features
from kplex_pool.data import NDPDataset, CustomDataset

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import accuracy_score
from tqdm import tqdm



class TestScoring:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.y_true = [y for _, y in test_dataset]
    
    def __call__(self, net, X=None, y=None):
        y_pred = net.predict(self.test_dataset)

        return accuracy_score(self.y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoverPool')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--jumping_knowledge', type=str, default='cat')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--max_k', type=int, default=8)
    parser.add_argument('--k_step_factor', type=float, default=1.)
    parser.add_argument('--q', type=float, default=None)
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--easy', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--max_layers', type=int, default=3)
    parser.add_argument('--inner_layers', type=int, default=2)
    parser.add_argument('--to_pickle', type=str, default='cv_results.pickle')
    parser.add_argument('--from_pickle', type=str, default=None)
    args = parser.parse_args()

    if args.from_pickle is not None:
        results = pd.read_pickle(args.from_pickle)
        outer = results[(results.cv_type == 'outer') & results.valid_acc_best]
        outer.index = range(len(outer))
        test_acc = outer.test_acc[outer.groupby('fold').epoch.idxmax()]

        print("\nAccuracy: {:.2f} ± {:.2f}\n".format(
            100.*np.mean(test_acc), 
            100.*np.std(test_acc)
        ))

        exit()

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cpu'

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = 'cuda' 
    
    if args.dataset == 'NDPDataset':
        train, val, test = (NDPDataset('data/', 
                                       split=key, 
                                       easy=args.easy, 
                                       small=args.small) 
                            for key in ['train', 'val', 'test'])
        train_stop = len(train)
        val_stop = train_stop + len(val)
        test_stop = val_stop + len(test)
        train_idx = np.arange(train_stop)
        val_idx = np.arange(train_stop, val_stop)
        test_idx = np.arange(val_stop, test_stop)

        dataset = CustomDataset(list(train) + list(val) + list(test))
        skf_pbar = tqdm([(train_idx, test_idx)], disable=True) 
            
        X = np.arange(len(dataset)).reshape((-1, 1))
        y = dataset.data.y.numpy()

        benchmark = True
    else:
        dataset = TUDataset(root='data/' + args.dataset, name=args.dataset)

        if dataset.data.x is None:
            dataset = add_node_features(dataset)
            
        X = np.arange(len(dataset)).reshape((-1, 1))
        y = dataset.data.y.numpy()

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        skf_pbar = tqdm(list(skf.split(X, y)), leave=True, desc='Outer CV')
        sss_split = 1./(args.folds - 1)
        benchmark = False
    
    results = []
    test_acc = []

    shared_params = {
        'module': getattr(model, args.model), 
        'module__dataset': dataset,
        'module__dropout': args.dropout,
        'module__num_inner_layers': args.inner_layers,
        'module__jumping_knowledge': args.jumping_knowledge,
        'module__device': device,
        'batch_size': args.batch_size,
        'criterion': model.PoolLoss if args.model == 'DiffPool' else torch.nn.modules.loss.NLLLoss,
        'optimizer': torch.optim.Adam,
        'optimizer__weight_decay': 1e-4,
        'callbacks__print_log__sink': skf_pbar.write,
        'iterator_train__shuffle': True,
        'device': device
    }

    param_grid = {
        'optimizer__lr': [1e-3, 5e-4, 2e-4, 1e-4],
        'module__graph_sage': [True, False],
        'module__hidden': [64, 128],
        'module__num_layers': list(range(2, args.max_layers + 1))
    }

    if args.model == 'CoverPool':
        cover_fs = dict()
        kplex_cover = KPlexCover()
        shared_params.update(module__dense=args.dense)
        param_grid.update(module__k=2**np.arange(np.log2(args.max_k) + 1).astype(int))
    elif args.model == 'EdgePool':
        param_grid.update({
            'module__method': ['softmax', 'tanh'],
            'module__edge_dropout': [0.0, 0.2]
        })
    elif args.model != 'BaseModel':
        param_grid.update({
            'module__ratio': [0.25, 0.5, 0.75]
        })

    for out_iter, (train_idx, test_idx) in enumerate(skf_pbar):
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]

        if benchmark:
            val_X = X[val_idx]
            val_y = y[val_idx]
        else:
            in_sss = StratifiedShuffleSplit(n_splits=1, test_size=sss_split, random_state=42)
            train_idx, val_idx = next(in_sss.split(train_X, train_y))

            val_X = train_X[val_idx]
            val_y = train_y[val_idx]
            train_X = train_X[train_idx]
            train_y = train_y[train_idx]
        
        valid_ds = Dataset(val_X, val_y)
        test_ds = Dataset(test_X, test_y)

        gs_pbar = tqdm(list(ParameterGrid(param_grid)), leave=True, desc='Grid Search')
        test_scoring = TestScoring(test_ds)

        best_acc = 0.
        best_params = None

        for params in gs_pbar:
            gs_pbar.set_postfix({k.split('__')[1]: v for k, v in params.items()})

            if args.model == 'CoverPool':
                last_k = params.pop('module__k')
                ks = [last_k]

                for _ in range(2, params['module__num_layers']):
                    last_k = np.ceil(last_k*args.k_step_factor).astype(int)
                    ks.append(last_k)
                
                ks = tuple(ks)

                if ks not in cover_fs:
                    cover_fs[ks] = kplex_cover.get_cover_fun(ks, dataset, dense=args.dense, q=args.q, simplify=args.simplify)

                params['module__cover_fun'] = cover_fs[ks]
            
            net = NeuralNetClassifier(
                train_split=predefined_split(valid_ds), 
                max_epochs=args.max_epochs,
                callbacks=[('early_stopping', EarlyStopping)],
                callbacks__early_stopping__patience=args.patience,
                callbacks__early_stopping__sink=skf_pbar.write,
                **shared_params,
                **params
            ).fit(train_X, train_y)

            df = pd.DataFrame(net.history).drop('batches', 1)
            valid_acc = df[df.valid_acc_best].valid_acc.iloc[-1]

            df = pd.concat([df, pd.DataFrame([params for _ in df.iterrows()])], axis=1)
            df['cv_type'] = 'inner'
            df['fold'] = out_iter

            results.append(df)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_params = params
        
        gs_pbar.close()

        net = NeuralNetClassifier(
            train_split=predefined_split(valid_ds), 
            max_epochs=args.max_epochs,
            callbacks=[
                ('early_stopping', EarlyStopping),
                ('test_acc', EpochScoring)
            ],
            callbacks__early_stopping__patience=args.patience,
            callbacks__early_stopping__sink=skf_pbar.write,
            callbacks__test_acc__scoring=test_scoring,
            callbacks__test_acc__use_caching=False,
            callbacks__test_acc__lower_is_better=False,
            callbacks__test_acc__name='test_acc',
            **shared_params,
            **best_params
        ).fit(train_X, train_y)

        df = pd.DataFrame(net.history).drop('batches', 1)
        test_acc.append(df[df.valid_acc_best].test_acc.iloc[-1])

        df = pd.concat([df, pd.DataFrame([params for _ in df.iterrows()])], axis=1)
        df['cv_type'] = 'outer'
        df['fold'] = out_iter

        results.append(df)
        skf_pbar.set_postfix({
            'Accuracy': '{:.2f} ± {:.2f}\n'.format(
                    100.*np.mean(test_acc), 
                    100.*np.std(test_acc)
                )
            })
    
    skf_pbar.close()

    print('\nAccuracy: {:.2f} ± {:.2f}\n'.format(
        100.*np.mean(test_acc), 
        100.*np.std(test_acc)
    ))

    results = pd.concat(results, sort=False)

    if args.model == 'CoverPool':
        results = results.drop('module__cover_fun', 1)  # Ugly fix

    results.to_pickle(args.to_pickle)
