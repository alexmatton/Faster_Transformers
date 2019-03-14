#!/usr/bin/env bash

python train.py  --n_epochs 30  --save 1 --lr 1e-5 --weight_decay 0.0 --optimizer sgd --model transformer --flag transformer_full --max_source_positions 400 --max_target_positions 100


python train.py  --n_epochs 30  --save 0 --lr 1e-5 --weight_decay 0.0 --optimizer adam --model transformer --flag transformer_debug --max_source_positions 400 --max_target_positions 100 \
--data_path "datasets/cnn_debug"


CUDA_VISIBLE_DEVICES=0,1 fairseq-train  --task translation 'datasets/cnn_full_txt'  --max-source-positions 400 \
--max-target-positions 100 --max-tokens 10000 --raw-text -s src -t tgt -a transformer_iwslt_de_en --optimizer adam \
--label-smoothing 0.1 --dropout 0.3 --lr 0.0005 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 \
 --warmup-updates 4000 --warmup-init-lr '1e-07' \
 --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer --no-epoch-checkpoints --fp16 \
 --tensorboard-logdir 'checkpoints/tensorboard/transformer_full' --restore-file checkpoint_last.pt \
 --distributed-world-size 2

CUDA_VISIBLE_DEVICES=2 fairseq-generate 'datasets/cnn_full_txt' --task translation --max-source-positions 400 \
  --path checkpoints/transformer/checkpoint_best.pt --max-target-positions 100 --max-tokens 70000 --raw-text -s src -t tgt \
  --fp16 --beam 4  --num-workers 5 > test_output_epoch_18



CUDA_VISIBLE_DEVICES=3,4 fairseq-train  --task translation 'datasets/cnn_full_txt'  --max-source-positions 400 \
--max-target-positions 100 --max-tokens 10000 --raw-text -s src -t tgt -a lightconv-transformer-like --optimizer adam \
--label-smoothing 0.1 --dropout 0.3 --lr 0.0005 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 \
 --warmup-updates 4000 --warmup-init-lr '1e-07' \
 --adam-betas '(0.9, 0.98)' --save-dir checkpoints/lightweight --no-epoch-checkpoints --fp16 \
 --tensorboard-logdir 'checkpoints/tensorboard/lightweight_full' \
 --distributed-world-size 2 --encoder-conv-type lightweight --decoder-conv-type lightweight



CUDA_VISIBLE_DEVICES=0 fairseq-train  --task translation 'datasets/cnn_full_txt'  --max-source-positions 400 \
--max-target-positions 100 --max-tokens 8000 --raw-text -s src -t tgt -a local-transformer --optimizer adam \
--label-smoothing 0.1 --dropout 0.3 --lr 0.0005 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 \
 --warmup-updates 4000 --warmup-init-lr '1e-07' \
 --adam-betas '(0.9, 0.98)' --save-dir checkpoints/local_transformer --no-epoch-checkpoints --fp16 \
 --tensorboard-logdir 'checkpoints/tensorboard/local_transformer_full' --num-workers 10 \


CUDA_VISIBLE_DEVICES=1 fairseq-train  --task translation 'datasets/cnn_debug_txt'  --max-source-positions 400 \
 --max-target-positions 100 --max-tokens 8000 --raw-text -s src -t tgt -a local_transformer --optimizer adam \
  --label-smoothing 0.1 --dropout 0.3 --lr 0.0005 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay \
  0.0001 --criterion label_smoothed_cross_entropy --max-update 150000  --warmup-updates 4000 \
  --warmup-init-lr '1e-07'  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/local_transformer \
  --no-epoch-checkpoints --fp16   --num-workers 10 --kernel-size 400