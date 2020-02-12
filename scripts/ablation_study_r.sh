#!/bin/bash

DS=$1
ARGS="--min_layers=3 --max_layers=3 --dense --only_gcn --hidden=64 --dataset=${DS}"
DIR=./results/ablation_study/${DS}/

mkdir -p $DIR

FS=(8.0 4.0 2.0)

for K in 1 2 4; do
    for F in ${FS[@]}; do
        python -m benchmark.cv $ARGS --min_k=$K --max_k=$K --k_step_factor=$F --to_pickle=${DIR}/K${K}_F${F}.pickle
    done

    FS=("${FS[@]/$F}")
done
