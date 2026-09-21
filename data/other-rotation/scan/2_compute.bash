#!/usr/bin/env bash
module load gaussian/16-C.01  # lc4

for i in gaussian/*.com; do
  echo $i;
  g16 $i ;
done
