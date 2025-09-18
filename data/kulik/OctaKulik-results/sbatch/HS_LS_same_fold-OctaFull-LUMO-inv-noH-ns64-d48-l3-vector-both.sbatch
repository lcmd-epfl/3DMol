#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=HS_LS_same_fold-OctaFull-LUMO-vector
#SBATCH --array=0

FOLD=$SLURM_ARRAY_TASK_ID

TRAIN_FRAC=0.7995

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/kulik/OctaKulik-splits/HS_LS_same_fold_${size}/${{FOLD}}_test_indices.txt \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/izar/briling/cv \
--print_predictions \
--wandb_name cv10-HS_LS_same_fold-OctaFull-LUMO-inv-noH-ns64-d48-l3-vector-both \
--arch normal \
--classification False \
--dataset data/kulik/dataloader_kulik.py:OctaFull \
--distance_emb_dim 48 \
--dropout_p 0.1 \
--features torchchem_v1 \
--geometry default \
--features torchchem_v1 \
--graph_mode vector \
--invariant  \
--lr 0.0005 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode both \
--target_column LUMO \
--weight_decay 1e-05 \

