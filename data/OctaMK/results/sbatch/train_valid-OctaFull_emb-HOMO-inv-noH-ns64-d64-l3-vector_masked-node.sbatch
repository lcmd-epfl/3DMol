#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_valid-OctaFull_emb-HOMO-vector_masked
#SBATCH --array=0

FOLD=$SLURM_ARRAY_TASK_ID

TRAIN_FRAC=0.599

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-OctaMK-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/OctaMK/splits/train_valid_3612/${FOLD}_test_indices.txt \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name train_valid-OctaFull_emb-HOMO-inv-noH-ns64-d64-l3-vector_masked-node \
--arch normal \
--dataset data/OctaMK/dataloader_OctaMK.py:OctaFull \
--distance_emb_dim 64 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry default \
--features torchchem_v1 \
--graph_mode vector_masked \
--invariant  \
--lr 0.001 \
--max_neighbors 10 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 5.0 \
--seed 123 \
--sum_mode node \
--target_column HOMO \
--weight_decay 1e-05 \
--embedding_specs '{"multiplicity":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":101,"offset":0},"total_charge":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":201,"offset":100}}' 

