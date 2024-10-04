#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=LUMO-vector
#SBATCH --array=0-9
#SBATCH --exclude=i39

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/yuri/splits/fold_${FOLD}_test_indices.npy \
--logdir /scratch/izar/briling/cv \
--print_predictions \
--wandb_name cv10-yuri-LUMO-equiv-noH-ns64-nv32-d64-l3-vector-node \
--dataset yuri \
--distance_emb_dim 64 \
--dropout_p 0.1 \
--features torchchem_v1 \
--graph_mode vector \
--lr 0.001 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 32 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode node \
--target_column LUMO \
--train_frac 0.8 \
--weight_decay 1e-05 \

