#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=5:00:00
#SBATCH --job-name=QM9Rotation-rot633_sign-vector
#SBATCH --array=0-19

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
--num_epochs 128 \
--splitter random \
--logdir /scratch/briling/cv/ \
--print_predictions \
--wandb_name cv20-QM9Rotation-rot633_sign-pseudo-ns16-nv32-d16-both \
--arch pseudo \
--classification  \
--dataset data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation \
--distance_emb_dim 16 \
--dropout_p 0.0 \
--features torchchem_v1 \
--geometry dft \
--graph_mode vector \
--lr 0.0005 \
--max_neighbors 50 \
--n_conv_layers 3 \
--n_s 16 \
--n_v 32 \
--radius 5.0 \
--sum_mode both \
--train_frac 0.8 \
--weight_decay 0 \

