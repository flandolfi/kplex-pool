#!/bin/bash

POOL=CliquePool
ARGS="--model=${POOL} --max_layers=2"

for DS in COLLAB REDDIT-BINARY REDDIT-MULTI-5K; do
   python benchmark/cv.py $ARGS --dataset=$DS --to_pickle=results/${POOL}_${DS}.pickle --batch_size=1000
done

ARGS="$ARGS --dense"

for DS in IMDB-BINARY IMDB-MULTI; do
    python benchmark/cv.py $ARGS $OPT --dataset=$DS --to_pickle=results/${POOL}_${DS}.pickle
done