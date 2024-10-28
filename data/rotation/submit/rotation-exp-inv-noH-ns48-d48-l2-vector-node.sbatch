#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=inv-l2
#SBATCH --array=0
#SBATCH --exclude=i39

SEED=$(($SLURM_ARRAY_TASK_ID+666))

module purge
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-rotation-cv \
--CV 1 \
--num_epochs 512 \
--splitter random \
--logdir /scratch/izar/briling/cv \
--print_predictions \
--wandb_name cv10-rotation-exp-inv-l2-noH-ns48-d48-vector-node \
--dataset data/rotation/dataloader_rotation.py:Rotation \
--geometry xtb \
--distance_emb_dim 48 \
--dropout_p 0.05 \
--features torchchem_v1 \
--graph_mode vector \
--lr 0.001 \
--max_neighbors 50 \
--n_s 48 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed $SEED \
--sum_mode node \
--target_column specific_rotation \
--train_frac 0.8 \
--weight_decay 1e-05 \
--invariant  \
--n_conv_layers 2 \

