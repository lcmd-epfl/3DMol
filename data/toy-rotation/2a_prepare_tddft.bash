#/usr/bin/env bash

gdir=gaussian
sdir=/scratch/xe

mkdir -p $gdir

for i in xyz/[0-9]*.xyz; do
  echo $i;
  k=$(basename ${i::-4})
  j=$gdir/$k.com

  echo "%Mem=8GB
#P B3LYP/cc-pVDZ polar=optrot CPHF=RdFreq

Title Card Required

0 1" > $j
  tail -n +3 $i >> $j
  echo -e "\n589nm\n" >> $j

done
