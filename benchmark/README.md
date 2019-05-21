# Benchmarks #

## DD ##
```
python -m benchmark.eval --dataset=DD --epochs=50 --lr=0.01 --lr_decay_step=25
```

## PROTEINS ##
```
python -m benchmark.eval --dataset=PROTEINS --epochs=200 --lr=0.005
```

## ENZYMES ##
```
python -m benchmark.eval --dataset=ENZYMES --epochs=1000 --lr_decay_step=1000 --hidden=128 --lr=0.005
```




