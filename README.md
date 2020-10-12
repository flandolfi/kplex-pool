# K-Plex Cover Pooling for Graph Neural Networks #

This repository contains the implementation and the experimental setup used in 
*"K-Plex Cover Pooling for Graph Neural Networks"* (NeurIPS 2020).

## Abstract ##

We introduce a novel pooling technique which borrows from classical results in graph theory that is non-parametric and 
generalizes well to graphs of different nature and connectivity pattern. Our pooling method, named KPlexPool, builds on 
the concepts of graph covers and *k*-plexes, i.e. pseudo-cliques where each node can miss up to *k* links. The 
experimental evaluation on benchmarks on molecular and social graph classification shows that KPlexPool achieves state 
of the art performances against both parametric and non-parametric pooling methods in the literature, despite 
generating pooled graphs based solely on topological information.

## Contents ##

| Folder        | Description                                                |
|:--------------|:-----------------------------------------------------------|
| `benchmark/`  | Implementation of the benchmark tests (model cross-validation). 
| `kplex_pool/` | Python interface of the pooling module. 
| `cpu/`        | Implementation of the covering, the edge-pooling, and the graph simplification algorithms, as C++ pyTorch extensions.
| `scripts/`    | Bash scripts used to perform batch of experiments. Use these script to reproduce the results that can be found in the article.
| `test/`       | Test suite for debugging.

## Usage ##

### Requirements ###

To use this package make sure to have Python 3.7, CUDA 10.1 and GCC >=5.4.
We also suggest to use Anaconda/Miniconda.

### Installation ###

Run the following commands in the project directory:

```shell script
conda create -n kplex-pool python=3.7 -y
conda activate kplex-pool
conda install -y -c nvidia -c rapidsai -c numba -c conda-forge -c defaults \
      dask-cudf dask-cuda cugraph=0.15 cudatoolkit=10.1 c-compiler cxx-compiler

pip install -r requirements.txt
pip install .
``` 

Optionally, run `python setup.py test` to execute the test suite.

### Benchmark Tests ###

You can cross-validate or evaluate the following models (implemented in `benchmark/model.py`):
 
 - `KPlexPool` (named `CoverPool` in the code, for generalization purposes);
 - `DiffPool`;
 - `gPool` (named `TopKPool`);
 - `SAGPool`;
 - `EdgePool`;
 - `Graclus`.

#### Cross-Validation ####

To run an experiment, use the command `python -m benchmark.cv <...>`, with the following options
```
usage: cv.py [-h]
             [-m {BaseModel,CoverPool,DiffPool,TopKPool,SAGPool,EdgePool,Graclus}]
             [-d DS] [--jumping_knowledge {cat,lstm,}]
             [--global_pool_op POOL [POOL ...]]
             [--node_pool_op POOL [POOL ...]]
             [--edge_pool_op {add,max,min,mean}] [--max_epochs E] [--min_k K]
             [--max_k K] [-r R] [-q Q] [--simplify] [--dense] [--dense_from L]
             [--easy] [--small] [--only_gcn] [--patience PATIENCE] [-b B]
             [--dropout P] [--folds FOLDS] [-c H] [--min_layers L]
             [--max_layers L] [--inner_layers L] [--to_pickle PATH]
             [--from_pickle PATH]

Cross-validate a given model.

optional arguments:
  -h, --help            show this help message and exit
  -m {BaseModel,CoverPool,DiffPool,TopKPool,SAGPool,EdgePool,Graclus}, --model {BaseModel,CoverPool,DiffPool,TopKPool,SAGPool,EdgePool,Graclus}
                        Model to cross-validate (default: CoverPool).
  -d DS, --dataset DS   Dataset on which the cross-validation is performed.
                        Must be a dataset from the TU Dortmund collection or
                        NPDDataset (default: PROTEINS).
  --jumping_knowledge {cat,lstm,}
                        Jumping knowledge aggregation type (default: cat).
  --global_pool_op POOL [POOL ...]
                        Global aggregation function(s) (default: ['add']).
  --node_pool_op POOL [POOL ...]
                        Local aggregation functions(s) (default: ['add']).
  --edge_pool_op {add,max,min,mean}
                        Edge weight aggregation function (default: add)
  --max_epochs E        Number of maximum epochs per training (default: 1000).
  --min_k K             Left bound of the log-scale (base 2) k-parameter
                        space. Only applicable to CoverPool (default: 1).
  --max_k K             Right bound of the log-scale (base 2) k-parameter
                        space. Only applicable to CoverPool (default: 8).
  -r R, --k_step_factor R
                        Reduction factor of the k parameter. Only applicable
                        to CoverPool (default: 1.0).
  -q Q, --q Q           Hub-promotion quantile threshold (must be a float in
                        [0, 1]). Only applicable to CoverPool (default: None).
  --simplify            Apply simplification to coarsened grpahs. Only
                        applicable to CoverPool (default: False).
  --dense               Use the dense form computation (default: False).
  --dense_from L        Use the dense form starting from the given layer, and
                        use the sparse form for the other layers. Only
                        applicable to BaseModel and CoverPool (default: 0).
  --easy                Easy dataset. Only applicable to NPDDataset (default:
                        False).
  --small               Small dataset. Only applicable to NPDDataset (default:
                        False).
  --only_gcn            Do not use SAGEConv in the grid search (default:
                        False).
  --patience PATIENCE   Early-stopping patience epochs (default: 20).
  -b B, --batch_size B  The size of the batches used during training(default:
                        -1).
  --dropout P           Dropout probability in the final dense layer (default:
                        0.3).
  --folds FOLDS         Number of outer folds (default: 10).
  -c H, --hidden H      Fix the number of channels during the grid search.
  --min_layers L        Minimum number of layers in the grid search(default:
                        2).
  --max_layers L        Maximum number of layers in the grid search (default:
                        3).
  --inner_layers L      Number of layers within each convolutional block
                        (default: 2).
  --to_pickle PATH      Path of the output pickle storing the history of the
                        cross-validation (default: cv_results.pickle).
  --from_pickle PATH    Compute the outer-fold accuracy of the given history.
                        If set, ignores every other parameter and does not
                        perform cross validation (default: None).
```

##### Example: #####

    python -m benchmark.cv \
        --dense \
        --model CoverPool \
        --dataset ENZYMES \
        --k_step_factor 0.5 \
        --node_pool_op add max

#### Evaluation ####

To run an experiment, use the command `python -m benchmark.eval <...>`, with the following options
```
usage: eval.py [-h]
               [-m {BaseModel,CoverPool,DiffPool,TopKPool,SAGPool,EdgePool,Graclus}]
               [-d DS] [--jumping_knowledge {cat,lstm,}]
               [--global_pool_op POOL [POOL ...]]
               [--node_pool_op POOL [POOL ...]]
               [--edge_pool_op {add,max,min,mean}] [--epochs E] [-k K] [-r R]
               [-q Q] [--simplify] [--dense] [--dense_from L] [--easy]
               [--small] [-b B] [--dropout P] [-c H] [-l L] [--inner_layers L]
               [--cover_priority {random,min_degree,max_degree,min_uncovered,max_uncovered,default}]
               [--kplex_priority {random,min_degree,max_degree,min_uncovered,max_uncovered,min_in_kplex,max_in_kplex,min_candidates,max_candidates,default}]
               [--lr LR] [--weight_decay WD] [--ratio RATIO] [--split S]
               [--method {softmax,sigmoid,tanh}] [--edge_dropout P]
               [--graph_sage] [--skip_covered] [--no_readout] [--no_cache]
               [--ks [K [K ...]]]

Evaluate a given model.

optional arguments:
  -h, --help            show this help message and exit
  -m {BaseModel,CoverPool,DiffPool,TopKPool,SAGPool,EdgePool,Graclus}, --model {BaseModel,CoverPool,DiffPool,TopKPool,SAGPool,EdgePool,Graclus}
                        Model to evaluate (default: CoverPool).
  -d DS, --dataset DS   Dataset on which the cross-validation is performed.
                        Must be a dataset from the TU Dortmund collection or
                        NPDDataset (default: PROTEINS).
  --jumping_knowledge {cat,lstm,}
                        Jumping knowledge aggregation type (default: cat).
  --global_pool_op POOL [POOL ...]
                        Global aggregation function(s) (default: ['add']).
  --node_pool_op POOL [POOL ...]
                        Local aggregation functions(s) (default: ['add']).
  --edge_pool_op {add,max,min,mean}
                        Edge weight aggregation function (default: add)
  --epochs E            Number of epochs (default: 1000).
  -k K, --k K           K-plex parameter. Only applicable to CoverPool
                        (default: 2).
  -r R, --k_step_factor R
                        Reduction factor of the k parameter. Only applicable
                        to CoverPool (default: 1.0).
  -q Q, --q Q           Hub-promotion quantile threshold (must be a float in
                        [0, 1]). Only applicable to CoverPool (default: 1).
  --simplify            Apply simplification to coarsened grpahs. Only
                        applicable to CoverPool.
  --dense               Use the dense form computation.
  --dense_from L        Use the dense form starting from the given layer, and
                        use the sparse form for the other layers. Only
                        applicable to BaseModel and CoverPool (default: 0).
  --easy                Easy dataset. Only applicable to NPDDataset.
  --small               Small dataset. Only applicable to NPDDataset.
  -b B, --batch_size B  The size of the batches used during training (default:
                        -1).
  --dropout P           Dropout probability in the final dense layer (default:
                        0.3).
  -c H, --hidden H      Number of channels (default: 64).
  -l L, --layers L      Number of convolutional blocks
  --inner_layers L      Number of layers within each convolutional block
                        (default: 2).
  --cover_priority {random,min_degree,max_degree,min_uncovered,max_uncovered,default}
                        Priority used to extract the pivot node (default:
                        default).
  --kplex_priority {random,min_degree,max_degree,min_uncovered,max_uncovered,min_in_kplex,max_in_kplex,min_candidates,max_candidates,default}
                        Priority used to extract the next k-plex candidate
                        node (default: default).
  --lr LR               Learning rate (default: 0.001).
  --weight_decay WD     Weight decay (default: 0.001).
  --ratio RATIO         Output/input number of nodes ratio. Only Applicable to
                        DiffPool, TopKPool, and SAGPool (default: 0.8).
  --split S             Test split (default: 0.1).
  --method {softmax,sigmoid,tanh}
                        Function to apply to compute the edge score from raw
                        edge scores. Only applicable to EdgePool (default:
                        softmax).
  --edge_dropout P      probability with which to drop edge scores during
                        training. Only applicable to EdgePool (default: 0.2).
  --graph_sage          Use SAGEConv instead of GCNConv.
  --skip_covered        Give max priority to uncovered nodes. Only applicable
                        to CoverPool
  --no_readout          Use only the final global pooling aggregation as input
                        to the dense layers.
  --no_cache            Do not precoumpute the graph covers.
  --ks [K [K ...]]      Specify the k value for each layer. Only applicable to
                        CoverPool. If set, --k, --k_factor and --layers
                        options will be ignored.

```

##### Example #####

    python -m benchmark.eval \
        --dense \
        --lr 0.001 \
        --epochs 100 \
        --hidden 128 \
        --model CoverPool \
        --dataset ENZYMES \
        --k_step_factor 0.5 \
        --node_pool_op add max

### Experiment Scripts ###

#### Cross-Validation Scripts ####
Run `./scripts/cv_`*`<EXP>`*`.sh` from the project folder to execute the experiments reported in the article.

#### Ablation Study Scripts ####
Run `./scripts/ablation_study_[p|r].sh `*`<DS>`* to execute an ablation study on `KPlexPool` either on the `p` 
parameter (called `q` above, since it refers to the *quantile* threshold), or on the reduction factor `r`, using `DS` 
as dataset.

#### Results ####

##### Chemical Datasets #####

| Model                 | DD        | ENZYMES   | NCI1      | PROTEINS  |
|:----------------------|:---------:|:---------:|:---------:|:---------:|
| Baseline              | 74.79     | 43.00     | 78.08     | 71.61     |
| Graclus               | 77.42     | 42.67     | 78.06     | 74.12     |
| TopKPool              | 73.35     | 39.17     | 74.09     | 74.12     |
| SAGPool               | 74.75     | 37.67     | 78.01     | 73.31     |
| DiffPool              | OOR       | **46.00** | 76.76     | 75.02     |
| KPlexPool             | **77.76** | 39.67     | **79.17** | 75.11     |
| KPlexPool (*f* = 0.5) | 75.98     | 43.33     | 78.09     | **75.92** |

##### Social Datasets #####

| Model                 | COLLAB    | IDMB-B    | IDMB-M    | REDDIT-B  | REDDIT-5K |
|:----------------------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Baseline              | 74.44     | 69.20     | 47.20     | 84.85     | 51.71     |
| Graclus               | 72.80     | 68.70     | 47.20     | 86.85     | **53.77** |
| TopKPool              | 73.30     | 68.20     | 46.93     | 78.60     | 50.33     |
| SAGPool               | 73.40     | 65.40     | 46.33     | 80.15     | 49.79     |
| DiffPool              | 70.92     | 68.80     | 47.07     | OOR       | OOR       |
| KPlexPool             | **76.20** | **72.00** | 46.60     | 86.45     | 50.65     |
| KPlexPool (*p* = 95)  | 75.98     | 69.40     | **48.73** | **87.90** | 51.37     |

