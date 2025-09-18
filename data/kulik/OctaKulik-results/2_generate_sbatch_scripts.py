#!/usr/bin/env python3

from glob import glob
import numpy as np

for config_file in glob('configs/config-*-*-*-????????-????????.dat'):

    for split_type in ['random', 'HS_LS_same_fold', 'train_valid']:

        if split_type == 'HS_LS_same_fold' and short_dataset == 'OctaLow':
            continue

        skip_keys = ['train_frac']
        base_config = np.loadtxt(config_file, skiprows=1, dtype=str)
        base_config = {key: val.strip('"') for key, val in base_config}
        base_config = {key: val if val not in ['nan', 'None'] else '0' for key, val in base_config.items()}
        config = '\n'.join([f'--{key if key != "graph_method" else "features"} {val if val.lower()!="true" else ""} \\' for key, val in base_config.items() if key not in skip_keys])

        short_dataset = base_config['dataset'].split(':')[-1]
        short_name = f"{split_type}-{short_dataset}-{base_config['target_column']}-{'inv' if base_config['invariant'].lower()=='true' else 'equiv'}-noH"
        full_name = f"{short_name}-ns{base_config['n_s']}-{'nv'+base_config['n_v']+'-' if base_config['n_v']!='0' else ''}d{base_config['distance_emb_dim']}-l{base_config['n_conv_layers']}-{base_config['graph_mode']}-{base_config['sum_mode']}"


        if short_dataset == 'OctaFull':
            size = 3612
        elif short_dataset == 'OctaLow':
            size = 1806
        if split_type == 'random':
            splitter = f'test:data/kulik/OctaKulik-splits/random_{size}/${{FOLD}}_test_indices.txt'
            array = '0-9'
            if short_dataset=='OctaFull':
                train_frac_bash = 'if (( $FOLD > 1 )); then TRAIN_FRAC=0.8; else TRAIN_FRAC=0.7995; fi'
            else:
                train_frac_bash = 'if (( $FOLD > 5 )); then TRAIN_FRAC=0.801; else TRAIN_FRAC=0.8; fi'
        elif split_type == 'train_valid':
            splitter = f'test:data/kulik/OctaKulik-splits/train_valid_{size}/${{FOLD}}_test_indices.txt'
            array = '0'
            train_frac_bash = 'TRAIN_FRAC=0.599'
        elif split_type == 'HS_LS_same_fold':
            splitter = f'test:data/kulik/OctaKulik-splits/HS_LS_same_fold_{size}/${{FOLD}}_test_indices.txt'
            array = '0'
            train_frac_bash = 'TRAIN_FRAC=0.7995'

        header=f"""#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name={split_type}-{short_dataset}-{base_config['target_column']}-{base_config['graph_mode']}
#SBATCH --array={array}

FOLD=$SLURM_ARRAY_TASK_ID

{train_frac_bash}

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \\"""

        run_config = f"""--device cuda \\
--experiment_name 3DMol-yuri-cv \\
--CV 1 \\
--num_epochs 512 \\
--splitter {splitter} \\
--train_frac ${{TRAIN_FRAC}} \\
--logdir /scratch/briling/cv/ \\
--print_predictions \\
--wandb_name cv10-{full_name} \\"""

        with open(f'sbatch/{full_name}.sbatch', 'w') as f:
            print("\n".join((header, run_config, config)), file=f)
            print(file=f)
