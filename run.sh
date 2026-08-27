#wandb enabled
#wandb disabled

head -n 101 data/qm9-rotation/data.csv > data/qm9-rotation/data_mini.csv

python train.py --device='cuda' \
                --experiment_name='run-gpu3' \
                --noH \
                --wandb_name test-run \
                --n_conv_layers 3 \
                --num_epochs=8 \
                --dataset data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation \
                --print_predictions \
                --dataloader_args 'csv_name:data_mini.csv' \
                #--arch pseudo \
                #--subset 16 \
                #--train_frac 0.8004 \
                #--train_frac 0.8 \
                #--train_frac 0.798 \
                #--target_column specific_rotation \
                #--xtb \
                #--splitter test:data/yuri/splits/fold_0_test_indices.npy \
                #--subset 100 \
                #--target_column U0_Hartree \
                #--dataset qm9 \
                #--num_epochs=5 \
                #--dataset test \
                #--dataset proparg \
                #--dropout_p 0.0 \
                #--graph_mode vector
                #--process \
                #--learning_curve \
                #--splitter sizeasc \
                #--splitter scaffold \
                #--dataset proparg \
                #--seed 130 \
#                --checkpoint logs/run-gpu/231006-135410.579171-xe.best_checkpoint.pt \
                #--rxnmapper \
                #--process \
