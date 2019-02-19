#!/bin/bash

num_runs=$1
for it in {1..$num_runs}
do
    python test_sdae-3layer.py --lr 0.0001 --pretrainepochs 300 --epochs 300 --save model/sdae-run-$it.pt --name gmmvae-run-$it
    python test_gmmvae-3layer.py --lr 0.0001 --lr-stepwise 0.01 --alpha 0.1 --epochs 300 --pretrain model/sdae-run-$it.pt --save model/gmmvae-run-$it.pt --name gmmvae-run-$it
done
