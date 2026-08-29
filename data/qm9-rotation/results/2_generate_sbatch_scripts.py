#!/usr/bin/env python3

from glob import glob
import numpy as np

seeds = ' '.join(str(i) for i in range(666, 676))

for config_file in glob('configs/config-*-*-*-????????-????????.dat'):

    skip_keys = ['target_column', 'seed', 'arch']
    base_config = np.loadtxt(config_file, skiprows=1, dtype=str)
    base_config = {key: val.strip('"') for key, val in base_config}
    base_config = {key: val if val not in ['nan', 'None'] else '0' for key, val in base_config.items()}
    config = '\n'.join([f'--{key} {val if val.lower()!="true" else ""} \\' for key, val in base_config.items() if (key not in skip_keys and val.lower() != 'false')])

    target_column = base_config['target_column']
    for target in ['rot589', 'rot589_sign', 'rot589_abs']:

        short_dataset = base_config['dataset'].split(':')[-1]

        for arch in ['normal', 'normal_scaled', 'pseudo_scaled', 'pseudo_nonscaled', 'both_scaled', 'both_nonscaled']:

            short_name = f"{short_dataset}-{target}-{arch}"
            full_name = f"{short_name}-ns{base_config['n_s']}-{'nv'+base_config['n_v']}-d{base_config['distance_emb_dim']}"

            header=f"""#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=00:59:59
#SBATCH --job-name={target}-{arch}

        conda activate 3dmol

        for SPLIT in `seq 0 9`; do

        SEED=$((SPLIT+666))

        python train.py \\"""

            run_config = f"""--device cuda \\
        --experiment_name 3DMol-rotation-cv \\
        --project 3dmol-rot \\
        --CV 1 \\
        --seed $SEED \\
        --target_column {target} {'--classification' if 'sign' in target else ''} \\
        --arch {arch} \\
        --num_epochs 128 \\
        --patience 32 \\
        --max_gap 0.05 \\
        --splitter "test:data/qm9-rotation/splits/test.$SPLIT.dat;val:data/qm9-rotation/splits/val.$SPLIT.dat" \\
        --logdir cv/ \\
        --print_predictions \\
        --wandb_name cv10-{full_name} \\"""

            tail = """&
            sleep 5
            done"""

            with open(f'sbatch/{full_name}.bash', 'w') as f:
                print("\n".join((header, run_config, config, tail)), file=f)
                print(file=f)
