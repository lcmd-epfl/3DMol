#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=00:59:59
#SBATCH --job-name=specific_rotation_computed_sign-pseudo_nonscaled

        if [[ "$HOSTNAME" == "newstarrebornberlin" ]]; then conda activate 3dmol; else conda activate equireact-kuma; fi

        for SPLIT in `seq 0 9`; do

        SEED=$((SPLIT+666))

        python train.py \
--dataset data/reaxys-rotation/dataloader_rotation.py:Rotation \
--device cuda \
        --experiment_name 3DMol-rotation-cv \
        --project 3dmol-rot \
        --seed $SEED \
        --target_column specific_rotation_computed_sign --classification \
        --arch pseudo_nonscaled \
        --num_epochs 32 \
        --max_gap 0.05 \
        --splitter "test:data/reaxys-rotation/splits/test.$SPLIT.dat;val:data/reaxys-rotation/splits/val.$SPLIT.dat" \
        --logdir cv/ \
        --print_predictions \
        --wandb_name cv10-miniRXOR-specific_rotation_computed_sign-pseudo_nonscaled_tanh_noH-ns48-nv32-d64 \
--noH \
--distance_emb_dim 64 \
--dropout_p 0 \
--features torchchem_v1 \
--geometry dft \
--graph_mode vector \
--lr 0.0005 \
--n_conv_layers 3 \
--n_s 48 \
--n_v 32 \
--optimizer AdamW \
--radius 5 \
--train_frac 0.8 \
--weight_decay 0.0001 \

            done

