# K-Plex Cover Pooling for Graph Neural Networks #

This repository contains the implementation and the experimental setup used in *"K-Plex Cover Pooling for Graph Neural Networks"* (ICML2020).

## Abstract ##

Graph pooling methods provide mechanisms for structure reduction that are intended to ease the diffusion of context between nodes further in the graph, and that typically leverage community discovery mechanisms or node and edge pruning heuristics. In this paper, we introduce a novel pooling technique which borrows from classical results in graph theory that is non-parametric and generalizes well to graphs of different nature and connectivity pattern. Our pooling method, named KPlexPool, builds on the concepts of graph covers and *k*-plexes, i.e. pseudo-cliques where each node can miss up to *k* links. The experimental evaluation on benchmarks on molecular and social graph classification shows that KPlexPool achieves state of the art performances against both parametric and non-parametric pooling methods in the literature, despite generating pooled graphs based solely on topological information.

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

To use this package make sure to have Python (>=3.6) and also the following packages
    
    numpy>=1.16.1
    torch>=1.1.0
    torch-cluster>=1.4.2
    torch-geometric>=1.3.0
    torch-geometric-benchmark>=0.1.0
    torch-scatter>=1.2.0
    torch-sparse>=0.4.0
    tqdm>=4.31.1

To run the benchmark tests, install also `scikit-learn` and `skorch`.

### Installation ###

Run `python setup.py install`. Optionally, run `python setup.py test` to execute the test suite.

### Benchmark Tests ###

You can perform cross-validation on the following models (implemented in `benchmark/model.py`):
 
 - `KPlexPool` (named `CoverPool` in the code, for generalization purposes);
 - `DiffPool`;
 - `gPool` (named `TopKPool`);
 - `SAGPool`;
 - `EdgePool`;
 - `Graclus`.

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

#### Example: ####

    python -m benchmark.cv \
        --dense \
        --model CoverPool \
        --dataset ENZYMES \
        --k_step_factor 0.5 \
        --node_pool_op add max

### Experiment Scripts ###

#### Cross-Validation Scripts ####
Run `./scripts/cv_`*`<EXP>`*`.sh` from the project folder to execute the experiments reported in the article.

#### Ablation Study Scripts ####
Run `./scripts/ablation_study_[p|r].sh `*`<DS>`* to execute an ablation study on `KPlexPool` either on the `p` parameter (called `q` above, since it refers to the *quantile* threshold), or on the reduction factor `r`, using `DS` as dataset.
