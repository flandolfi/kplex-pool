import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product

import torch

from torch_geometric.datasets import TUDataset

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from benchmark import model

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KPlexPool')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--jumping_knowledge', type=str, default='cat')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--inner_folds', type=int, default=10)
    parser.add_argument('--outer_folds', type=int, default=10)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--inner_layers', type=int, default=2)
    parser.add_argument('--to_pickle', type=str, default='cv_results.pickle')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    dataset = TUDataset(root='data/' + args.dataset, name=args.dataset)
    X = np.arange(len(dataset)).reshape((-1, 1))
    y = dataset.data.y.numpy()

    out_skf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=42)
    out_pbar = tqdm(list(out_skf.split(X, y)), leave=True, position=0, desc='Outer CV')
    results = []
    test_acc = []

    shared_params = {
        'module': getattr(model, args.model), 
        'module__dataset': dataset,
        'module__num_layers': args.layers,
        'module__dropout': args.dropout,
        'module__num_inner_layers': args.inner_layers,
        'module__jumping_knowledge': args.jumping_knowledge,
        'batch_size': args.batch_size,
        'criterion': model.PoolLoss if args.model == 'DiffPool' else torch.nn.modules.loss.NLLLoss,
        'optimizer': torch.optim.Adam,
        'callbacks__print_log__sink': out_pbar.write,
        'iterator_train__shuffle': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    param_grid = {
        'lr': [5e-3], #, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4],
        'optimizer__weight_decay': [1e-2], #, 1e-3, 1e-4, 1e-5],
        'module__graph_sage': [True], #, False],
        'module__hidden': [64], #, 128]
    }

    if args.model == 'KPlexPool':
        param_grid.update({
            'module__k': [[i, j] for i, j in product([1], [1])] #, 2, 4, 8], [1, 2])]
        })
    elif args.model == 'EdgePool':
        param_grid.update({
            'module__method': ['softmax'], #, 'tanh'],
            'module__edge_dropout': [0.0] #, 0.1, 0.2, 0.3]
        })
    else:
        ratios = [0.25] #, 0.5, 0.75]
        param_grid.update({
            'module__ratio': [[i, j] for i, j in product(ratios, ratios)]
        })

    for out_iter, (out_train_idx, out_test_idx) in enumerate(out_pbar):
        out_train_X = X[out_train_idx]
        out_train_y = y[out_train_idx]
        out_test_X = X[out_test_idx]
        out_test_y = y[out_test_idx]
        gs_pbar = tqdm(list(ParameterGrid(param_grid)), leave=True, position=1, desc='Grid Search')

        best_acc = 0.
        best_epoch = 0.
        best_params = None

        for params in gs_pbar:
            in_skf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=42)
            in_pbar = tqdm(list(in_skf.split(out_train_X, out_train_y)), leave=True, position=2, desc='Inner CV')
            gs_pbar.set_postfix(params)
            best_valid_acc = []
            best_epoch = []

            for in_iter, (in_train_idx, in_val_idx) in enumerate(in_pbar):
                in_train_X = X[in_train_idx]
                in_train_y = y[in_train_idx]
                in_val_X = X[in_val_idx]
                in_val_y = y[in_val_idx]

                valid_ds = Dataset(in_val_X, in_val_y)
                net = NeuralNetClassifier(
                    train_split=predefined_split(valid_ds), 
                    max_epochs=args.max_epochs,
                    callbacks=[('early_stopping', EarlyStopping)],
                    callbacks__early_stopping__patience=args.patience,
                    callbacks__early_stopping__sink=out_pbar.write,
                    **shared_params,
                    **params
                ).fit(in_train_X, in_train_y)

                df = pd.DataFrame(net.history).drop('batches', 1)
                best_valid_acc.append(df[df.valid_acc_best].valid_acc.iloc[-1])
                best_epoch.append(df[df.valid_acc_best].epoch.iloc[-1])

                df = pd.concat([df, pd.DataFrame([params for _ in df.iterrows()])], axis=1)
                df['cv_type'] = 'inner'
                df['inner_fold'] = in_iter
                df['outer_fold'] = out_iter

                results.append(df)

            in_pbar.close()

            mean_acc = np.mean(best_valid_acc)

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_epoch = int(np.median(best_epoch))
                best_params = params
        
        gs_pbar.close()

        test_ds = Dataset(out_test_X, out_test_y)
        net = NeuralNetClassifier(
            train_split=predefined_split(test_ds), 
            max_epochs=best_epoch,
            **shared_params,
            **best_params
        ).fit(out_train_X, out_train_y)

        test_acc.append(net.history[-1, 'valid_acc']) # The last one, not the best one
        df = pd.DataFrame(net.history).drop('batches', 1)

        df = pd.concat([df, pd.DataFrame([params for _ in df.iterrows()])], axis=1)
        df['cv_type'] = 'outer'
        df['outer_fold'] = out_iter

        results.append(df)
    
    out_pbar.close()
    pd.concat(results, sort=False).to_pickle(args.to_pickle)

    print("\nAccuracy: {:.2f} ± {:.2f}".format(
        100.*np.mean(test_acc), 
        100.*np.std(test_acc)
    ))
