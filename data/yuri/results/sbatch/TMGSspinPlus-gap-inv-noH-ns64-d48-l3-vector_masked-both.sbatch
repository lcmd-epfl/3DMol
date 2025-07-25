#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=TMGSspinPlus-gap-vector_masked
#SBATCH --array=0-9
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/yuri/splits/${FOLD}_test_indices.txt \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-TMGSspinPlus-gap-inv-noH-ns64-d48-l3-vector_masked-both \
--arch normal \
--dataset data/yuri/dataloader_yuri.py:TMGSspinPlus \
--distance_emb_dim 48 \
--dropout_p 0.1 \
--features torchchem_v1 \
--geometry default \
--graph_mode vector_masked \
--invariant  \
--lr 0.001 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode both \
--target_column gap \
--train_frac 0.8 \
--weight_decay 1e-05 \

