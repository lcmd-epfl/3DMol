#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=05:00
#SBATCH --job-name=tmPHOTO-gap
#SBATCH --array=0-4
#SBATCH --qos=debug
##SBATCH --partition=h100
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

for i in data/kulik/tmPHOTO-xyz/*.xyz; do
        md5sum $i;
done

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-timing \
--CV 1 \
--num_epochs 128 \
--dataloader_args "csv_path:data/kulik/tmPHOTO_gap_500_${FOLD}.csv" \
--logdir /scratch/briling/cv \
--print_predictions \
--wandb_name time-tmPHOTO-gap-inv-noH-ns64-d64-l3-energy-node \
--dataset data/kulik/dataloader_kulik.py:tmPHOTO \
--distance_emb_dim 64 \
--dropout_p 0.05 \
--features torchchem_v1 \
--geometry xtb \
--graph_mode energy \
--invariant  \
--lr 0.001 \
--max_neighbors 25 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed 123 \
--sum_mode node \
--target_column gap \
--train_frac 0.8 \
--weight_decay 1e-05 \
--process \

