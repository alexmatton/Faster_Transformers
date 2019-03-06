#!/usr/bin/env bash

python train.py  --n_epochs 30  --save 1 --lr 1e-5 --weight_decay 0.0 --optimizer sgd --model transformer --flag transformer_full --max_source_positions 400 --max_target_positions 100