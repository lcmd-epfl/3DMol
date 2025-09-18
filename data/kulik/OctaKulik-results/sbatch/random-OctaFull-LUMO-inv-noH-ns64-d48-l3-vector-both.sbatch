#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=random-OctaFull-LUMO-vector
#SBATCH --array=0-9

FOLD=$SLURM_ARRAY_TASK_ID

if (( $FOLD > 1 )); then TRAIN_FRAC=0.8; else TRAIN_FRAC=0.7995; fi

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/kulik/OctaKulik-splits/random_$3612/${FOLD}_test_indices.txt \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-random-OctaFull-LUMO-inv-noH-ns64-d48-l3-vector-both \
--arch normal \
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

