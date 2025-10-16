#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --job-name=rot589_abs-normal
#SBATCH --array=0-19

FOLD=$SLURM_ARRAY_TASK_ID
if (( $FOLD > 9 )) ; then SEED=$((666 + $FOLD - 10)) ; else SEED=$((123 + $FOLD)) ; fi

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name 3DMol-rotation-cv \
--CV 1 \
--seed $SEED \
--target_column rot589_abs \
--num_epochs 128 \
--splitter random \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv20-QM9Rotation-rot589_abs-normal-ns32-nv0-d64-node \
--arch normal \
--dataset data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation \
--distance_emb_dim 64 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry dft \
--graph_mode vector \
--lr 0.001 \
--max_neighbors 10 \
--n_conv_layers 3 \
--n_s 32 \
--n_v None \
--radius 5.0 \
--sum_mode node \
--train_frac 0.8 \
--weight_decay 0 \
--invariant \

