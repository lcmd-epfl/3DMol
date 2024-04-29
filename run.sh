wandb enabled
wandb disabled
python train.py --device='cuda' \
                --experiment_name='run-gpu3' \
                --num_epochs=5 \
                --atom_mapping \
                --noH \
                --xtb_subset \
                --wandb_name test-run \
                --train_frac 0.8 \
                --dataset homometric \
                --seed 123 \
                --invariant \
                --n_conv_layers 3 \
                #--subset 100 \
                #--learning_curve \
                #--splitter sizeasc \
                #--splitter yasc \
                #--splitter scaffold \
                #--dataset proparg \
                #--seed 130 \
#                --checkpoint logs/run-gpu/231006-135410.579171-xe.best_checkpoint.pt \
                #--rxnmapper \
                #--process \
