#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=tmSCO-gap-vector
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
--experiment_name 3DMol-yuri-cv \
--CV 1 \
--num_epochs 512 \
--splitter test:data/kulik/tmSCO-splits/tmSCO_${FOLD}_test_indices.npy \
--train_frac ${TRAIN_FRAC} \
--logdir /scratch/briling/cv \
--print_predictions \
--wandb_name cv10-tmSCO_emb-gap-inv-noH-ns64-d32-l3-vector-both \
--dataset data/kulik/dataloader_kulik.py:tmSCO \
--distance_emb_dim 32 \
--dropout_p 0.05 \
--features torchchem_v1 \
--geometry xtb \
--features torchchem_v1 \
--graph_mode vector \
--invariant  \
--lr 0.0005 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 5.0 \
--seed 123 \
--sum_mode both \
--target_column gap \
--weight_decay 1e-05 \
--embedding_specs '{"multiplicity":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":101,"offset":0},"total_charge":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":201,"offset":100}}' 

