#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name=random-OctaLow_emb-splitting-vector_masked
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
--wandb_name cv10-random-OctaLow_emb-splitting-inv-noH-ns64-d32-l3-vector_masked-both \
--arch normal \
--dataset data/kulik/dataloader_kulik.py:OctaLow \
--distance_emb_dim 32 \
--dropout_p 0 \
--features torchchem_v1 \
--geometry default \
--features torchchem_v1 \
--graph_mode vector_masked \
--invariant  \
--lr 0.0005 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode both \
--target_column splitting \
--weight_decay 1e-05 \
--embedding_specs '{"multiplicity":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":101,"offset":0},"total_charge":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":1024,"num_classes":201,"offset":100}}' 

