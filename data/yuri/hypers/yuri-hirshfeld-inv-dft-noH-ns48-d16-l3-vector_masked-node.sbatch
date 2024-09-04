#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=hirshfeld-vector_masked
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
--wandb_name cv10-yuri-hirshfeld-inv-noH-ns48-d16-l3-vector_masked-node \
--dataset yuri \
--distance_emb_dim 16 \
--dropout_p 0.0 \
--features torchchem_v1 \
--graph_mode vector_masked \
--invariant  \
--lr 0.001 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 48 \
--n_v 0 \
--noH  \
--radius 10.0 \
--seed 123 \
--sum_mode node \
--target_column hirshfeld \
--train_frac 0.8 \
--weight_decay 1e-05 \

