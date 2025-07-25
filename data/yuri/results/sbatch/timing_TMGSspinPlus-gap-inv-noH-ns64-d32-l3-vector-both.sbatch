#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=TMGSspinPlus-gap-vector
#SBATCH --array=0-4
#SBATCH --qos=debug        # so they run in order
##SBATCH --partition=h100
#SBATCH --partition=l40s

SEED=$((666+$SLURM_ARRAY_TASK_ID))

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

for i in data/yuri/0-XYZS/*.xyz; do
        md5sum $i;
done

python train.py \
--device cuda \
--experiment_name 3DMol-yuri-timing \
--CV 1 \
--num_epochs 128 \
--splitter random
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name time-TMGSspinPlus-gap-inv-noH-ns64-d32-l3-vector-both \
--arch normal \
--dataset data/yuri/dataloader_yuri.py:TMGSspinPlus \
--distance_emb_dim 32 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry default \
--graph_mode vector \
--invariant  \
--lr 0.001 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 0 \
--noH  \
--radius 2.5 \
--seed $SEED \
--sum_mode both \
--target_column gap \
--train_frac 0.8 \
--weight_decay 1e-05 \
--process \

