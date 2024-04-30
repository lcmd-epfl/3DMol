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
                --subset 100 \
                --dataset qm9 \
                #--dataset dsC7O2H10nsd \
                #--num_epochs=128 \
                #--num_epochs=5 \
                #--dataset test \
                #--dataset proparg \
                #--dropout_p 0.0 \
                #--graph_mode vector
                #--process \
                #--learning_curve \
                #--splitter sizeasc \
                #--splitter yasc \
                #--splitter scaffold \
                #--dataset proparg \
                #--seed 130 \
#                --checkpoint logs/run-gpu/231006-135410.579171-xe.best_checkpoint.pt \
                #--rxnmapper \
                #--process \
