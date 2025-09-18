#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_valid-OctaFull-HOMO-vector
#SBATCH --array=0

FOLD=$SLURM_ARRAY_TASK_ID

TRAIN_FRAC=0.599

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/kulik/OctaKulik-splits/train_valid_$3612/${FOLD}_test_indices.txt \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/izar/briling/cv \
--print_predictions \
--wandb_name cv10-train_valid-OctaFull-HOMO-inv-noH-ns48-d64-l3-vector-node \
--arch normal \
--dataset data/kulik/dataloader_kulik.py:OctaFull \
--distance_emb_dim 64 \
--dropout_p 0.05 \
--features torchchem_v1 \
--geometry default \
--features torchchem_v1 \
--graph_mode vector \
--invariant  \
--lr 0.001 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 48 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode node \
--target_column HOMO \
--weight_decay 1e-05 \

