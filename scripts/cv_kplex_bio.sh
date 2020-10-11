#!/bin/bash

POOL=${POOL}

for KF in 1.0 0.5; do
   DS=ENZYMES
   python benchmark/cv.py --dense --model=${POOL} --dataset=$DS --k_step_factor=$KF --to_pickle=results/${POOL}_F${KF}_${DS}_ADDMAX.pickle --node_pool_op add max

   for DS in DD PROTEINS NCI1; do
      python benchmark/cv.py --model=${POOL} --dataset=$DS --k_step_factor=$KF --to_pickle=results/${POOL}_F${KF}_${DS}_ADDMAX.pickle --node_pool_op add max
   done
done

