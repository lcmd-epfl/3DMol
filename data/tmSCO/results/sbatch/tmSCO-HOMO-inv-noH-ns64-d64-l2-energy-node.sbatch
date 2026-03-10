#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=tmSCO-HOMO-energy
#SBATCH --array=0-9

FOLD=$SLURM_ARRAY_TASK_ID

TRAIN_FRAC=0.798
if [ $FOLD != 0 ]; then TRAIN_FRAC=$(printf "%.1f" $TRAIN_FRAC); fi

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-tmSCO-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/tmSCO/splits/tmSCO_${FOLD}_test_indices.npy \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/briling/cv \
--print_predictions \
--wandb_name cv10-tmSCO-HOMO-inv-noH-ns64-d64-l2-energy-node \
--dataset data/tmSCO/dataloader_tmSCO.py:tmSCO \
--distance_emb_dim 64 \
--dropout_p 0.1 \
--features torchchem_v1 \
--geometry xtb \
--features torchchem_v1 \
--graph_mode energy \
--invariant  \
--lr 0.001 \
--max_neighbors 25 \
--n_conv_layers 2 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode node \
--target_column HOMO \
--weight_decay 1e-05 \

