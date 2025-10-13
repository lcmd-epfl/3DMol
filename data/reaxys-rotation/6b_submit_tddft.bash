source ~/soft/modules/module-load-g16
for i in gaussian/*.com; do
  if [ ! -f ${i/.com/.log} ]; then
    sbatch --job-name $(basename $i) --cpus-per-task=1 --mem=8GB --wrap "g16 $i";
  fi
done
