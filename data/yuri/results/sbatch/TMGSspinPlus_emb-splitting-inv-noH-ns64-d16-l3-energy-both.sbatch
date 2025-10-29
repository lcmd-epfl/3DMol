#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=TMGSspinPlus-splitting-energy
#SBATCH --array=0-9
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/yuri/splits/${FOLD}_test_indices.txt \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-TMGSspinPlus_emb-splitting-inv-noH-ns64-d16-l3-energy-both \
--arch normal \
--dataset data/yuri/dataloader_yuri.py:TMGSspinPlus \
--distance_emb_dim 16 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry default \
--graph_mode energy \
--invariant  \
--lr 0.001 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 5.0 \
--seed 123 \
--sum_mode both \
--target_column splitting \
--train_frac 0.8 \
--weight_decay 1e-05 \
--embedding_specs '{"multiplicity":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":101,"offset":0},"total_charge":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":201,"offset":100}}' \

