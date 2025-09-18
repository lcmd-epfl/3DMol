#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=random-OctaLow-splitting-energy
#SBATCH --array=0-9

FOLD=$SLURM_ARRAY_TASK_ID

if (( $FOLD > 5 )); then TRAIN_FRAC=0.801; else TRAIN_FRAC=0.8; fi

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/kulik/OctaKulik-splits/random_1806/${FOLD}_test_indices.txt \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-random-OctaLow-splitting-inv-noH-ns48-d48-l3-energy-both \
--arch normal \
--dataset data/kulik/dataloader_kulik.py:OctaLow \
--distance_emb_dim 48 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry default \
--features torchchem_v1 \
--graph_mode energy \
--invariant  \
--lr 0.001 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 48 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode both \
--target_column splitting \
--weight_decay 1e-05 \

