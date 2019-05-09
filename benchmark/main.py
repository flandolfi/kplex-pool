from itertools import product
import sys
import argparse

from kernel.datasets import get_dataset
from kernel.train_eval import cross_validation_with_val_set

from kernel.top_k import TopK
from kernel.diff_pool import DiffPool
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool

from .kplex_pool import KPlexPool, KPlexPoolPre, KPlexPoolPost, KPlexPoolPreKOE

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=25)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

layers = [2, 3] # 4, 5]
hiddens = [64]
ks = [1, 4, 16, 64]
datasets = ['DD', 'PROTEINS', 'COLLAB',] #, 'IMDB-BINARY', 'REDDIT-BINARY', 'ENZYMES', 
nets = [
    # TopK,
    # DiffPool,
    # Set2SetNet,
    # SortPool,
    KPlexPool, 
    KPlexPoolPre, 
    KPlexPoolPreKOE, 
    KPlexPoolPost
]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Loss: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, loss, val_loss, test_acc), file=sys.stderr)

results = []

for dataset_name, Net in product(datasets, nets):
    param_it = product(layers, hiddens, ks) if Net.__name__.startswith('KPlexPool') else product(layers, hiddens)
    best_result = (float('inf'), 0, 0)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))

    for params in param_it:
        dataset = get_dataset(dataset_name, sparse=Net is not DiffPool)
        model = Net(*((dataset,) + params))
        model_desc = "L: {}, H: {}".format(*params[:2])

        if Net.__name__.startswith('KPlexPool'):
            model_desc += ", K: {}".format(params[2])
        
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
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
   
print('-----\n{}'.format('\n'.join(results)))
