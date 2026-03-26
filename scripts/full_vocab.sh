#!/bin/bash
# Set PYTHONPATH so Python can find the 'src' folder
export PYTHONPATH=$PYTHONPATH:$(pwd)

export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

export TMPDIR="/vol/tensusers6/vbolzonella/cache"
mkdir -p $TMPDIR

mkdir -p "vol/tensusers6/vbolzonella/mbart-expanded/dict"
mkdir -p "vol/tensusers6/vbolzonella/mbart-expanded/akkademia"

mkdir -p "vol/tensusers6/vbolzonella/mbart-expanded/maas"

# python3 -m src.training.mbart \
#   --training_mode "supervised" \
#   \
#   --model_type "mbart-expanded" \
#   --internal_train_data_path "dataset/training_input/dictionary" \
#   --save_model_path "/vol/tensusers6/vbolzonella/models/mbart-expanded-v2/dict" \
#   --model_name "mbart-expanded-vocab-expansion-dict" \
#   \
#   --save_model_every 3 \
#   --learning_rate 1e-4 \
#   --max_epochs 25 \
#   --precision "32" \
#   --early_stopping_patience 100 \
#   \
#   --lora_r 32 \
#   --lora_alpha 16 \
#   \
#   --arabic_init \
#   --diacritic_mode \
#   \
#   --gpu_num 4 \
#   --batch_size 80 \
#   --max_length 15 \
#   --train_data_ratio 1 \
#   --checkpoint_monitor "train_loss"
#
# sleep 30

python3 -m src.training.mbart \
  --training_mode "supervised" \
  \
  --model_type "mbart-expanded" \
  --checkpoint_path "/vol/tensusers6/vbolzonella/models/mbart-expanded-v2/dict/metric=train_loss=0.194-epoch=14-15-15h.ckpt" \
  --internal_train_data_path "dataset/training_input/external" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/mbart-expanded-v2/akkademia" \
  --model_name "mbart-expanded-vocab-expansion-akk" \
  \
  --learning_rate 1e-4 \
  --save_model_every 3 \
  --max_epochs 65 \
  --precision "32" \
  --early_stopping_patience 4 \
  \
  --lora_r 32 \
  --lora_alpha 16 \
  \
  --arabic_init \
  --diacritic_mode \
  \
  --gpu_num 4 \
  --checkpoint_monitor "geo" \
  --eval_every 3 \
  --batch_size 40 \
  --max_length 150

sleep 30

python3 -m src.training.mbart \
  --training_mode "supervised" \
  \
  --model_type "mbart-expanded" \
  --checkpoint_path "/vol/tensusers6/vbolzonella/models/mbart-expanded-v2/akkademia/last.ckpt" \
  --internal_train_data_path "dataset/training_input/internal" \
  --save_model_path "/vol/tensusers6/vbolzonella/models/mbart-expanded-v2/maas" \
  --model_name "mbart-expanded-vocab-expansion-maas" \
  \
  --save_model_every 3 \
  --learning_rate 1e-4 \
  --max_epochs 65 \
  --precision "32" \
  --early_stopping_patience 4 \
  \
  --lora_r 32 \
  --lora_alpha 16 \
  \
  --arabic_init \
  --diacritic_mode \
  \
  --gpu_num 4 \
  --batch_size 8 \
  --checkpoint_monitor "geo" \
  --eval_every 3 \
  --batch_size 8
