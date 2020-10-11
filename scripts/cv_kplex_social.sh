#!/bin/bash

POOL=CoverPool
ARGS="--model=${POOL} --max_layers=2 --max_k=1"

for DS in REDDIT-BINARY REDDIT-MULTI-5K; do
   python benchmark/cv.py $ARGS --dataset=$DS --to_pickle=results/${POOL}_Q1.00_${DS}.pickle --batch_size=20 --dense --dense_from=1
   python benchmark/cv.py $ARGS --dataset=$DS --q=0.95 --to_pickle=results/${POOL}_Q0.95_${DS}.pickle --batch_size=2000
done

ARGS="$ARGS --dense"

for DS in IMDB-BINARY IMDB-MULTI COLLAB; do
    if [ $DS = COLLAB ]; then
        OPT="--batch_size=1000"
    else
        OPT=""
    fi

    python benchmark/cv.py $ARGS $OPT --dataset=$DS --to_pickle=results/${POOL}_Q1.00_${DS}.pickle
    python benchmark/cv.py $ARGS $OPT --dataset=$DS --q=0.95 --to_pickle=results/${POOL}_Q0.95_${DS}.pickle
done