#!/usr/bin/env python3

from glob import glob
import numpy as np

for config in glob('configs/config-*-*-*-????????-????????.dat'):

    skip_keys = ['graph_method']
    base_config = np.loadtxt(config, skiprows=1, dtype=str)
    base_config = {key: val.strip('"') for key, val in base_config}
    base_config = {key: val if val not in ['nan', 'None'] else '0' for key, val in base_config.items()}
    config = '\n'.join([f'--{key if key != "graph_method" else "features"} {val if val.lower()!="true" else ""} \\' for key, val in base_config.items() if key not in skip_keys])

    short_dataset = base_config['dataset'].split(':')[-1]
    short_name = f"{short_dataset}-{base_config['target_column']}-{'inv' if base_config['invariant'].lower()=='true' else 'equiv'}-noH"
    full_name = f"{short_name}-ns{base_config['n_s']}-{'nv'+base_config['n_v']+'-' if base_config['n_v']!='0' else ''}d{base_config['distance_emb_dim']}-l{base_config['n_conv_layers']}-{base_config['graph_mode']}-{base_config['sum_mode']}"


    header=f"""#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name={short_dataset}-{base_config['target_column']}-{base_config['graph_mode']}
#SBATCH --array=0-9
#SBATCH --partition=l40s

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \\"""

    run_config = f"""--device cuda \\
--experiment_name 3DMol-yuri-cv \\
--CV 1 \\
--num_epochs 512 \\
--splitter test:data/yuri/splits/${{FOLD}}_test_indices.txt \\
--logdir /scratch/briling/cv/ \\
--print_predictions \\
--wandb_name cv10-{full_name} \\"""

    with open(f'sbatch/{full_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config)), file=f)
        print(file=f)
