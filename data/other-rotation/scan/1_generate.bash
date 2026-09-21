#!/usr/bin/env bash

TETR=109.47122063449069  # acos(-1/3) * 180.0 / pi
V=v  # or use `vmol` from `pip install vmol`

for i in `seq 0 23` ; do
  OUT=rot$(printf '%02d' $i)
  sed s/XXX/$((15*$i))/ template.in | sed s/TETR/${TETR}/ > ${OUT}.in
  ${V} ${OUT}.in gui:0 com:z > xyz/${OUT}.xyz

  {
  echo '# CAM-B3LYP/6-31G** polar=optrot CPHF=RdFreq'
  echo
  echo $i
  echo
  echo 0 1
  tail xyz/${OUT}.xyz -n +3
  echo
  echo 355nm,589.3nm,633nm
  } > gaussian/${OUT}.com

  rm ${OUT}.in

done

