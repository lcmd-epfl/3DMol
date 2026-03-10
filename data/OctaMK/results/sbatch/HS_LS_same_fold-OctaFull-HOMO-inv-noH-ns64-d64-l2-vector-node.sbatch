#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=HS_LS_same_fold-OctaFull-HOMO-vector
#SBATCH --array=0-9

FOLD=$SLURM_ARRAY_TASK_ID

if (( $FOLD > 5 )); then TRAIN_FRAC=0.8004; else TRAIN_FRAC=0.7995; fi

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-OctaMK-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/OctaMK/splits/HS_LS_same_fold_3612/${FOLD}_test_indices.txt \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-HS_LS_same_fold-OctaFull-HOMO-inv-noH-ns64-d64-l2-vector-node \
--arch normal \
--dataset data/OctaMK/dataloader_OctaMK.py:OctaFull \
--distance_emb_dim 64 \
--dropout_p 0.05 \
--features torchchem_v1 \
--geometry default \
--features torchchem_v1 \
--graph_mode vector \
--invariant  \
--lr 0.001 \
--max_neighbors 10 \
--n_conv_layers 2 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode node \
--target_column HOMO \
--weight_decay 1e-05 \

