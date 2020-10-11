#!/bin/bash

DS=$1
ARGS="--min_layers=3 --max_layers=3 --dense --only_gcn --hidden=64 --dataset=${DS}"
DIR=./results/ablation_study/${DS}/

mkdir -p $DIR

FS=(1.000 0.500 0.250 0.125)

for K in 8 4 2 1; do
    for F in ${FS[@]}; do
        python benchmark/cv.py $ARGS --min_k=$K --max_k=$K --k_step_factor=$F --to_pickle=${DIR}/K${K}_F${F}.pickle
    done

    FS=("${FS[@]/$F}")
done
