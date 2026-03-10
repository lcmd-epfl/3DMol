#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --job-name=sweep
#SBATCH --array=0-11
#SBATCH --partition=l40s

PROPS=('HOMO' 'LUMO' 'gap' 'dipole_moment_Debye' 'splitting' 'hirshfeld')
PROP=$(($SLURM_ARRAY_TASK_ID/2))
PROP=${PROPS[$PROP]}

if [ $(( $SLURM_ARRAY_TASK_ID  % 2)) -eq 0 ]; then
  LOCAL='--local';
else
  LOCAL='';
fi

echo $PROP $LOCAL

module purge
conda activate equireact-kuma
python -c 'import torch; print(torch.cuda.is_available())'

wandb enabled
python sweep.py -t $PROP $LOCAL -d TMGSspinPlus
