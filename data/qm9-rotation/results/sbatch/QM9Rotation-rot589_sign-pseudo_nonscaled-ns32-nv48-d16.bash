#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=00:59:59
#SBATCH --job-name=rot589_sign-pseudo_nonscaled

        conda activate 3dmol

        for SPLIT in `seq 0 9`; do

        SEED=$((SPLIT+666))

        python train.py \
--device cuda \
        --experiment_name 3DMol-rotation-cv \
        --project 3dmol-rot \
        --CV 1 \
        --seed $SEED \
        --target_column rot589_sign --classification \
        --arch pseudo_nonscaled \
        --num_epochs 128 \
        --patience 32 \
        --max_gap 0.05 \
        --splitter "test:data/qm9-rotation/splits/test.$SPLIT.dat;val:data/qm9-rotation/splits/val.$SPLIT.dat" \
        --logdir cv/ \
        --print_predictions \
        --wandb_name cv10-QM9Rotation-rot589_sign-pseudo_nonscaled-ns32-nv48-d16 \
--dataset data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation \
--distance_emb_dim 16 \
--dropout_p 0 \
--features torchchem_v1 \
--geometry dft \
--graph_mode vector \
--lr 0.0005 \
--n_conv_layers 3 \
--n_s 32 \
--n_v 48 \
--optimizer AdamW \
--radius 5 \
--train_frac 0.8 \
--weight_decay 0 \
&
            sleep 5
            done

