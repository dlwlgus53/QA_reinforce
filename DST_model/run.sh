#!/bin/sh
#SBATCH -J trainWoz # Job name
#SBATCH -o  ./out/test.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A100 # queue  name  or  partiton name

#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:1
#SBTACH   --ntasks=1
#SBATCH   --tasks-per-node=16
#SBATCH     --mail-user=jihyunlee@postech.ac.kr
#SBATCH     --mail-type=ALL

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module purge
module add cuda/10.4
module add cuDNN/cuda/10.4/8.0.4.30
#module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA_new "
conda activate QA_new

export PYTHONPATH=.

python main.py \
    --save_prefix debugging \
    --do_train 1 \
    --do_short 1 \
    --seed 1 \
    --g 1\
    --max_epoch 3 \
    --train_path '../woz-data/MultiWOZ_2.1/train_data.json' \
    --dev_path '../woz-data/MultiWOZ_2.1/dev_data.json' \
    --test_path '../woz-data/MultiWOZ_2.1/test_data.json' \
    --max_length 512 \
    --batch_size 8 \
    --test_batch_size 16 \



