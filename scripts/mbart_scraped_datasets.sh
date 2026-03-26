#!/bin/bash
# Set PYTHONPATH so Python can find the 'src' folder
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TMPDIR="/vol/tensusers6/vbolzonella/cache"
mkdir -p $TMPDIR

python3 -m src.training.mbart \
  --model_type "mbart" \
  --checkpoint_path "/vol/tensusers6/vbolzonella/models/full-mbart/dict/metric=train_loss=0.229-epoch=26-17-12h.ckpt" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/scraped/" \
  --model_name "scraped" \
  \
  --dataset_configs \
  "dataset/training_input/external:internal:supervised:0.7" \
  "dataset/training_input/external:internal:self_supervised:0.15" \
  "dataset/training_input/external/only_self_supervised:internal:self_supervised:0.15" \
  \
  --save_model_every 3 \
  --learning_rate 1e-4 \
  --max_epochs 200 \
  --precision "32" \
  --early_stopping_patience 10 \
  --checkpoint_monitor "val_loss" \
  --dropout 0.35 \
  --attention_dropout 0.35 \
  \
  --normalize_chars \
  \
  --batch_size 60 \
  --gpu_num 4 \
  --max_length 150 \
  --train_data_ratio 0.95 \
  --eval_every 3
