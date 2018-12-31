#!/bin/bash
python models.py --data-dir /home/gray/code/seqgan-opinion-spam/data/cache/ \
    --mode sample \
    --checkpoint-dir /tmp/chkpts/ \
    --model LanguageModelRNN \
    --RNN-sizes 128 \
    --embed-dim 64 \
    --seq-length 32 \
    --dataset-name AllChar \
    --dataset-type LM \
    --sample-seed-text "Those rats at the ho"\
    --sample-length 100 \
    --temperature 0.6

