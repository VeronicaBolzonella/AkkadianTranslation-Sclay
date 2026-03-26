#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --output=output.out
#SBATCH --time=12:00:00
#SBATCH --mail-user=analeopold
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/.bashrc

<
cd /vol/csedu-nobackup/course/I00041_informationretrieval/users/analeopold/MLiP---Translate-Akkadian-to-English---Kaggle-

export PYTHONPATH=$PTHONPATH:$(pwd)/src

source .venv/bin/activate

python -m src.training.byt5_main --output_dir "/vol/csedu-nobackup/course/I00041_informationretrieval/users/analeopold/models/byt5-akkadian-base"