#!/usr/bin/env python3

import numpy as np

for y in ['gap', 'HOMO', 'N_FOD', 'dipole', 'hirshfeld']: #, 'LUMO', 'splitting']:

    base_config = np.loadtxt(f'{y}-config.dat', skiprows=1, dtype=str)
    base_config = {key: val.strip('"') for key, val in base_config}
    config = '\n'.join([f'--{key} {val if val!="true" else ""} \\' for key, val in base_config.items()])

    short_name = f"{base_config['dataset']}-{base_config['target_column']}-{'inv' if base_config['invariant']=='true' else 'equiv'}-noH"
    full_name = f"{short_name}-ns{base_config['n_s']}-nv{base_config['n_v']}-d{base_config['distance_emb_dim']}-l{base_config['n_conv_layers']}-{base_config['graph_mode']}-{base_config['sum_mode']}"

    header=f"""#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --job-name=3dmol
#SBATCH --array=0-9
#SBATCH --exclude=i39

FOLD=$SLURM_ARRAY_TASK_ID

module purge
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \\"""

    run_config = f"""--device cuda \\
--experiment_name 3DMol-yuri-cv \\
--CV 1 \\
--num_epochs 512 \\
--splitter test:data/yuri/splits/fold_${{FOLD}}_test_indices.npy \\
--logdir /scratch/izar/briling/cv \\
--wandb_name cv-{full_name} \\"""

    with open(f'{short_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config)), file=f)
        print(file=f)
