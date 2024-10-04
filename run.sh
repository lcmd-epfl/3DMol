wandb enabled
wandb disabled
python train.py --device='cuda' \
                --experiment_name='run-gpu3' \
                --noH \
                --wandb_name test-run \
                --train_frac 0.8 \
                --seed 123 \
                --invariant \
                --n_conv_layers 2 \
                --num_epochs=8 \
                --dataset process/dataloader_rotation.py:Rotation \
                #--target_column specific_rotation \
                #--xtb \
                #--splitter test:data/yuri/splits/fold_0_test_indices.npy \
                #--splitter scaffold \
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
