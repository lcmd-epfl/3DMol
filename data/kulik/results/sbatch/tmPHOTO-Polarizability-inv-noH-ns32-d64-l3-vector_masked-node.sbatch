#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=tmPHOTO-Polarizability-vector_masked
#SBATCH --array=0-9
#SBATCH --exclude=i39

FOLD=$SLURM_ARRAY_TASK_ID

TRAIN_FRAC=0.8
if (( $FOLD > 7 )); then TRAIN_FRAC=${TRAIN_FRAC}004; fi

module purge
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/kulik/tmPHOTO-splits/tmPHOTO_${FOLD}_test_indices.npy \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/izar/briling/cv \
--print_predictions \
--wandb_name cv10-tmPHOTO-Polarizability-inv-noH-ns32-d64-l3-vector_masked-node \
--dataset data/kulik/dataloader_kulik.py:tmPHOTO \
--distance_emb_dim 64 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry xtb \
--features torchchem_v1 \
--graph_mode vector_masked \
--invariant  \
--lr 0.001 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 32 \
--n_v 0 \
--noH  \
--radius 5.0 \
--seed 123 \
--sum_mode node \
--target_column Polarizability \
--weight_decay 1e-05 \

