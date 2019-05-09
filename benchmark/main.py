from itertools import product
import sys
import argparse

from .datasets import get_dataset
from .train_eval import cross_validation_with_val_set
from .kplex_pool import KPlexPool, KPlexPoolPre, KPlexPoolPost, KPlexPoolPreKOE

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=25)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

layers = [2, 3] # 4, 5]
hiddens = [64]
ks = [1, 4, 16, 64]
datasets = ['PROTEINS'] #, 'IMDB-BINARY', 'REDDIT-BINARY', 'ENZYMES',  'COLLAB', 'DD'
nets = [
    KPlexPool, 
    KPlexPoolPre, 
    KPlexPoolPreKOE, 
    KPlexPoolPost
]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['val_acc']
    print('{:02d}/{:03d}: Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(
        fold, epoch, loss, val_loss, test_acc), file=sys.stderr)

results = []

for dataset_name, Net in product(datasets, nets):
    param_it = product(layers, hiddens, ks)
    best_result = (float('inf'), 0, 0)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))

    for params in param_it:
        dataset = get_dataset(dataset_name, sparse=True)
        model = Net(*((dataset,) + params))
        
        print("PARAMS: L: {}, H: {}, K: {}".format(*params))

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

    desc = '{:.4f} ± {:.4f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
   
print('-----\n{}'.format('\n'.join(results)))
