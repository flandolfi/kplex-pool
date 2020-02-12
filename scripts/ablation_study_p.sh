#!/bin/bash

DS=$1
ARGS="--dense --only_gcn --hidden=64 --batch_size=1000 --min_layers=2 --max_layers=2 --dataset=${DS}"
DIR=./results/ablation_study/${DS}/

mkdir -p $DIR

for K in 8 4 2 1; do
    python -m benchmark.cv $ARGS --min_k=$K --max_k=$K --to_pickle=${DIR}/K${K}_Q1.00.pickle

    for Q in 0.95 0.90 0.85 0.80; do
        python -m benchmark.cv $ARGS --min_k=$K --max_k=$K --q=$Q --to_pickle=${DIR}/K${K}_Q${Q}.pickle
    done
done
