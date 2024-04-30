#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=8:00:00
#SBATCH --job-name=equireact

module purge

SEED=666
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'

wandb enabled
python train.py --device='cuda' \
                --experiment_name=mol-test \
                --num_epochs=512 \
                --train_frac 0.8 \
                --seed ${SEED} \
                --logdir /scratch/izar/briling/nequimol \
                --CV 1 \
                --n_s=64 \
                --n_v=64 \
                --distance_emb_dim=32 \
                --radius=2.5 \
                --n_conv_layers=2 \
                --dropout_p=0.05 \
                --max_neighbors=10 \
                --sum_mode both \
                --dataset dsC7O2H10nsd \
                --wandb_name=test-dsC7O2H10nsd-inv-random-noH-ns64-nv64-d32-l2-vector-both \
                --graph_mode vector \
                --invariant \
                --noH \
