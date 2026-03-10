#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=TMGSspinPlus-dipole_moment_Debye-vector_masked
#SBATCH --array=0-9
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--project "3dmol-TMC-benchmark" \
--device cuda \
--experiment_name 3DMol-TMGSspinPlus-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/TMGSspinPlus/splits/${FOLD}_test_indices.txt \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv10-TMGSspinPlus_emb-dipole_moment_Debye-inv-noH-ns64-d16-l3-vector_masked-node \
--arch normal \
--dataset data/TMGSspinPlus/dataloader_TMGSspinPlus.py:TMGSspinPlus \
--distance_emb_dim 16 \
--dropout_p 0.1 \
--features torchchem_v1 \
--geometry default \
--graph_mode vector_masked \
--invariant  \
--lr 0.0005 \
--max_neighbors 10 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 10.0 \
--seed 123 \
--sum_mode node \
--target_column dipole_moment_Debye \
--train_frac 0.8 \
--weight_decay 1e-05 \
--embedding_specs '{"multiplicity":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":101,"offset":0},"total_charge":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":201,"offset":100}}' \

