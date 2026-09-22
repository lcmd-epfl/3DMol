#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=00:59:59
#SBATCH --job-name=rot589-both_nonscaled

        if [[ "$HOSTNAME" == "newstarrebornberlin" ]]; then conda activate 3dmol; else conda activate equireact-kuma; fi

        for SPLIT in `seq 0 9`; do

        SEED=$((SPLIT+666))

        python train.py \
--device cuda \
        --experiment_name 3DMol-rotation-cv \
        --project 3dmol-rot \
        --seed $SEED \
        --target_column rot589  \
        --arch both_nonscaled \
        --num_epochs 32 \
        --patience 16 \
        --gap_patience 16 \
        --max_gap 0.05 \
        --splitter "test:data/qm9-rotation/splits/test.aug.$SPLIT.dat;val:data/qm9-rotation/splits/val.aug.$SPLIT.dat" \
        --logdir cv/ \
        --print_predictions \
        --wandb_name cv10-QM9Rotation_aug-rot589-both_nonscaled_tanh-ns48-nv32-d64 \
--dataset data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation \
--dataloader_args 'csv_name:data_both.csv' \
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

