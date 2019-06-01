# Benchmarks #

## DD ##
```
python -m benchmark.eval --dataset=DD --epochs=50 --lr=0.001
```

## PROTEINS ##
```
python -m benchmark.eval --dataset=PROTEINS --epochs=25 --lr=0.0025
```

## ENZYMES ##
```
python -m benchmark.eval --dataset=ENZYMES --weight_decay=0 --lr=0.0001 --epochs=1000 --hidden=128 --layers=4 --k=4 --graph_sage
```




