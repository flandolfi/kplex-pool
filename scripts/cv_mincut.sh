#!/bin/bash

POOL=MinCutPool

for DS in ENZYMES NCI1 IMDB-BINARY IMDB-MULTI; do
    python benchmark/cv.py --model=$POOL --dataset=$DS --to_pickle=results/${POOL}_${DS}.pickle
done

for DS in PROTEINS COLLAB; do
    python benchmark/cv.py --model=$POOL --dataset=$DS --batch_size=256 --to_pickle=results/${POOL}_${DS}.pickle
done

for DS in DD REDDIT-BINARY REDDIT-MULTI-5K; do
    python benchmark/cv.py --model=$POOL --dataset=$DS --batch_size=6 --to_pickle=results/${POOL}_${DS}.pickle
done