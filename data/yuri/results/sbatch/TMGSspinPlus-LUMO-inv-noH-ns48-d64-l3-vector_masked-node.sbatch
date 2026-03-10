#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=TMGSspinPlus-LUMO-vector_masked
#SBATCH --array=0-9
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-TMGSspinPlus-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/TMGSspinPlus/splits/${FOLD}_test_indices.txt \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-TMGSspinPlus-LUMO-inv-noH-ns48-d64-l3-vector_masked-node \
--arch normal \
--dataset data/TMGSspinPlus/dataloader_TMGSspinPlus.py:TMGSspinPlus \
--distance_emb_dim 64 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry default \
--graph_mode vector_masked \
--invariant  \
--lr 0.0005 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 48 \
--n_v 0 \
--noH  \
--radius 5.0 \
--seed 123 \
--sum_mode node \
--target_column LUMO \
--train_frac 0.8 \
--weight_decay 0.0 \

