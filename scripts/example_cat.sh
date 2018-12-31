#!/bin/bash
python models.py --data-dir /home/gray/code/seqgan-opinion-spam/data/cache/ \
    --mode train \
    --checkpoint-dir /tmp/chkpts/ \
    --log-dir /tmp/log/ \
    --model LanguageModelRNN \
    --lr 0.001 \
    --RNN-sizes 128 \
    --embed-dim 64 \
    --seq-length 32 \
    --dataset-name AllChar \
    --dataset-type LM \
    --epochs 5 \
    --train-frac 0.7 \
    --test-frac 0.15 \
    --valid-frac 0.15 
