#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=tmPHOTO-dipole_moment_Debye-vector_masked
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
--wandb_name cv10-tmPHOTO-dipole_moment_Debye-inv-noH-ns32-d16-l3-vector_masked-both \
--dataset data/kulik/dataloader_kulik.py:tmPHOTO \
--distance_emb_dim 16 \
--dropout_p 0.1 \
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
--sum_mode both \
--target_column dipole_moment_Debye \
--weight_decay 1e-05 \

