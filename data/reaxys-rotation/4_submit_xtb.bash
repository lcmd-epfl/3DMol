for i in xyz/*.xyz; do sbatch --cpus-per-task=1 --mem=1GB --wrap "xtb $i --opt loose --cycles 1000 --namespace xyz-xtb/$(basename $i)"; done
