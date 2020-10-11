#!/bin/bash

for POOL in Louvain Leiden ECG Graclus; do
   for DS in IMDB-BINARY IMDB-MULTI; do
      python benchmark/cv.py --model=$POOL --dataset=$DS --to_pickle=results/${POOL}_${DS}.pickle
   done

   for DS in DD PROTEINS ENZYMES NCI1; do
       python benchmark/cv.py --model=$POOL --dataset=$DS --to_pickle=results/${POOL}_${DS}_ADDMAX.pickle --node_pool_op add max
   done

   for DS in COLLAB REDDIT-BINARY REDDIT-MULTI-5K; do
      python benchmark/cv.py --model=$POOL --dataset=$DS --batch_size=1000 --to_pickle=results/${POOL}_${DS}.pickle
   done
done

