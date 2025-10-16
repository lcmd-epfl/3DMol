#!/usr/bin/env python3

from glob import glob
import numpy as np

for config_file in glob('configs/config-*-*-*-????????-????????.dat'):

    skip_keys = ['target_column', 'seed']
    base_config = np.loadtxt(config_file, skiprows=1, dtype=str)
    base_config = {key: val.strip('"') for key, val in base_config}
    base_config = {key: val if val not in ['nan', 'None'] else '0' for key, val in base_config.items()}
    config = '\n'.join([f'--{key} {val if val.lower()!="true" else ""} \\' for key, val in base_config.items() if (key not in skip_keys and val.lower() != 'false')])

    target_column = base_config['target_column']
    for la in ['589', '633', '355']:

        target = target_column.replace('589', la)

        short_dataset = base_config['dataset'].split(':')[-1]
        short_name = f"{short_dataset}-{target}-{base_config['arch']}"
        full_name = f"{short_name}-ns{base_config['n_s']}-{'nv'+base_config['n_v']}-d{base_config['distance_emb_dim']}-{base_config['sum_mode']}"

        header=f"""#!/bin/bash -l
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=5:00:00
#SBATCH --job-name={short_dataset}-{target}-{base_config['graph_mode']}
#SBATCH --array=0-19

if (( $FOLD > 9 )) ; then SEED=$((666 + $FOLD - 10)) ; else SEED=$((123 + $FOLD)) ; fi

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \\"""

        run_config = f"""--device cuda \\
--experiment_name 3DMol-rotation-cv \\
--CV 1 \\
--seed $SEED \\
--num_epochs 128 \\
--splitter random \\
--logdir /scratch/briling/cv/ \\
--print_predictions \\
--wandb_name cv20-{full_name} \\"""

        with open(f'sbatch/{full_name}.sbatch', 'w') as f:
            print("\n".join((header, run_config, config)), file=f)
            print(file=f)
