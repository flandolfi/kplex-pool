from itertools import product
import sys
import argparse
import numpy as np

from benchmark.datasets import get_dataset
from benchmark.train_eval import cross_validation_with_val_set
from benchmark.models import KPlexPool, KPlexPoolSimplify

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--to_csv', type=str, default=None)
args = parser.parse_args()

layers = [2, 3, 4] 
hiddens = [64]
ks = [4, 8, 16]
datasets = ['PROTEINS'] #, 'IMDB-BINARY', 'REDDIT-BINARY', 'ENZYMES',  'COLLAB', 'DD'
nets = [
    KPlexPool, 
    # KPlexPoolSimplify
]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['val_acc']
    print('\r{:02d}/{:03d}: Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.3f}'.format(
                fold, epoch, loss, val_loss, test_acc), 
          end='' if epoch < args.epochs else '\n', 
          file=sys.stderr)

results = []

if args.to_csv is not None:
    fd = open(args.to_csv, 'w', buffering=1)
    fd.write("dataset,model,layers,units,k,acc_avg,acc_std\n")

for dataset_name, Net in product(datasets, nets):
    param_it = product(layers, hiddens, ks)
    best_result = (float('inf'), 0, 0)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    dataset = get_dataset(dataset_name, sparse=True)

    for params in param_it:
        dataset = get_dataset(dataset_name, sparse=True)
        model = Net(*((dataset,) + params))
        model_desc = "L: {}, H: {}, K: {}".format(*params)
        
        print("PARAMS: {}".format(model_desc))

        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=args.weight_decay,
            logger=logger if args.verbose else None)

        if args.to_csv is not None:
            fd.write("{},{},{},{},{},{:.3f},{:.3f}\n".format(dataset_name, Net.__name__,
                                                             *params, acc, std))

if args.to_csv is not None:
    fd.close()
