#!/bin/bash

POOL=DiffPool

for DS in ENZYMES NCI1 IMDB-BINARY IMDB-MULTI; do
    python benchmark/cv.py --model=$POOL --dataset=$DS --to_pickle=results/${POOL}_${DS}.pickle
done

for DS in PROTEINS COLLAB; do
    python benchmark/cv.py --model=$POOL --dataset=$DS --batch_size=256 --to_pickle=results/${POOL}_${DS}.pickle
done