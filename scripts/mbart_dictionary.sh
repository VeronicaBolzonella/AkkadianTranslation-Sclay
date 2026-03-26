#!/bin/bash

# Set PYTHONPATH so Python can find the 'src' folder
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TMPDIR="/vol/tensusers6/vbolzonella/cache"
mkdir -p $TMPDIR

# Launch the training using Accelerate
python3 -m src.training.mbart \
  --training_mode "supervised" \
  \
  --internal_train_data_path "dataset/training_input/internal" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/pretrained/dict" \
  --model_name "dict-pretrained" \
  \
  --save_model_every 10 \
  --learning_rate 1e-4 \
  --max_epochs 100 \
  --precision "32" \
  --early_stopping_patience 10 \
  \
  --lora_r 32 \
  --lora_alpha 16 \
  \
  --normalize_chars \
  --arabic_init \
  \
  --batch_size 80 \
  --gpu_num 4 \
  --max_length 15 \
  --train_data_ratio 1
