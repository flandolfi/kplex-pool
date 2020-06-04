#!/bin/bash

for POOL in BaseModel Graclus TopKPool SAGPool; do
   for DS in DD PROTEINS ENZYMES NCI1 IMDB-BINARY IMDB-MULTI; do
      python -m benchmark.cv --model=$POOL --dataset=$DS --to_pickle=results/${POOL}_${DS}.pickle
   done

   for DS in COLLAB REDDIT-BINARY REDDIT-MULTI-5K; do
      python -m benchmark.cv --model=$POOL --dataset=$DS --batch_size=1000 --to_pickle=results/${POOL}_${DS}.pickle
   done
done

POOL=Graclus

for DS in DD PROTEINS ENZYMES NCI1; do
    python -m benchmark.cv --model=$POOL --dataset=$DS --to_pickle=results/${POOL}_${DS}_ADDMAX.pickle --node_pool_op add max
done
