#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=TMGSspinPlus-LUMO-energy
#SBATCH --array=0-9
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--project "3dmol-TMC-benchmark" \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/yuri/splits/${FOLD}_test_indices.txt \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-TMGSspinPlus-LUMO-inv-noH-ns64-d32-l3-energy-both \
--arch normal \
--dataset data/yuri/dataloader_yuri.py:TMGSspinPlus \
--distance_emb_dim 32 \
--dropout_p 0.1 \
--features torchchem_v1 \
--geometry default \
--graph_mode energy \
--invariant  \
--lr 0.0005 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode both \
--target_column LUMO \
--train_frac 0.8 \
--weight_decay 0.001 \

