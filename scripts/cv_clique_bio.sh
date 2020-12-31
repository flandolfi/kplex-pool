#!/bin/bash

POOL=CliquePool
DS=ENZYMES

python benchmark/cv.py --dense --model=${POOL} --dataset=$DS --to_pickle=results/${POOL}_${DS}_ADDMAX.pickle --node_pool_op add max

for DS in DD PROTEINS NCI1; do
  python benchmark/cv.py --model=${POOL} --dataset=$DS --to_pickle=results/${POOL}_${DS}_ADDMAX.pickle --node_pool_op add max
done


