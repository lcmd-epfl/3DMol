#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --job-name=sweep
#SBATCH --array=0-11
##SBATCH --qos=debug
##SBATCH --partition=h100
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
python sweep.py -t $PROP $LOCAL -d yuri
#python sweep.py -t HOMO --local
#python sweep.py -t LUMO --local
#python sweep.py -t dipole_moment_Debye --local
#python sweep.py -t splitting --local
#python sweep.py -t hirshfeld --local --id y6iw3edy --num 1
#python sweep.py -t N_FOD --local
#python sweep.py -t N_FOD --id hp3mppyt --num 48
#python sweep.py -t N_FOD --local --id 62zfn7wz --num 14
#python sweep.py -t dipole_moment_Debye --local --id fcgcomyp --num 22

