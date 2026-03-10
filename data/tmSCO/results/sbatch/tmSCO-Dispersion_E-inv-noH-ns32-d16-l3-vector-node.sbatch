#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=tmSCO-Dispersion_E-vector
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
--wandb_name cv10-tmSCO-Dispersion_E-inv-noH-ns32-d16-l3-vector-node \
--dataset data/tmSCO/dataloader_tmSCO.py:tmSCO \
--distance_emb_dim 16 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry xtb \
--features torchchem_v1 \
--graph_mode vector \
--invariant  \
--lr 0.001 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 32 \
--n_v 0 \
--noH  \
--radius 5.0 \
--seed 123 \
--sum_mode node \
--target_column Dispersion_E \
--weight_decay 1e-05 \

