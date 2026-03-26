#!/bin/bash
# Set PYTHONPATH so Python can find the 'src' folder
export PYTHONPATH=$PYTHONPATH:$(pwd)

export TMPDIR="/vol/tensusers6/vbolzonella/cache"
mkdir -p $TMPDIR

mkdir -p "vol/tensusers6/vbolzonella/full-mbart/dict"

mkdir -p "vol/tensusers6/vbolzonella/full-mbart/akkademia"

mkdir -p "vol/tensusers6/vbolzonella/full-mbart/maas"

python3 -m src.training.mbart \
  --training_mode "supervised" \
  \
  --model_type "mbart" \
  --internal_train_data_path "dataset/training_input/dictionary" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/full-mbart/dict" \
  --model_name "full-mbart-dict" \
  \
  --save_model_every 3 \
  --learning_rate 1e-4 \
  --max_epochs 50 \
  --precision "32" \
  --early_stopping_patience 10 \
  \
  --lora_r 32 \
  --lora_alpha 16 \
  \
  --arabic_init \
  --normalize_chars \
  \
  --gpu_num 4 \
  --batch_size 80 \
  --max_length 15 \
  --train_data_ratio 1 \
  --checkpoint_monitor "train_loss"

python3 -m src.training.mbart \
  --training_mode "supervised" \
  \
  --model_type "mbart" \
  --checkpoint_path "/vol/tensusers6/vbolzonella/models/full-mbart/dict/last.ckpt" \
  --internal_train_data_path "dataset/training_input/external" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/full-mbart/akkademia" \
  --model_name "full-mbart-akk" \
  \
  --save_model_every 3 \
  --learning_rate 1e-4 \
  --max_epochs 100 \
  --precision "32" \
  --early_stopping_patience 10 \
  \
  --lora_r 32 \
  --lora_alpha 16 \
  \
  --arabic_init \
  --normalize_chars \
  \
  --gpu_num 4 \
  --batch_size 8 \
  --checkpoint_monitor "geo" \
  --eval_every 3 \
  --batch_size 60 \
  --max_length 150 \

python3 -m src.training.mbart \
  --training_mode "supervised" \
  \
  --model_type "mbart" \
  --checkpoint_path "/vol/tensusers6/vbolzonella/models/full-mbart/akkademia/last.ckpt" \
  --internal_train_data_path "dataset/training_input/internal" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/full-mbart/maas" \
  --model_name "full-mbart-maas" \
  \
  --save_model_every 3 \
  --learning_rate 1e-4 \
  --max_epochs 150 \
  --precision "32" \
  --early_stopping_patience 20 \
  \
  --lora_r 32 \
  --lora_alpha 16 \
  \
  --arabic_init \
  --normalize_chars \
  --name_swapping \
  \
  --gpu_num 4 \
  --batch_size 8 \
  --checkpoint_monitor "geo" \
  --eval_every 3 \
  --batch_size 8
