for i in `seq 0 23` ; do
  sed s/XXX/$((15*$i))/ template.in > rot$i.in
  echo 'z' | v rot$i.in gui:0 > xyz/rot$i.xyz

  {
  echo '# CAM-B3LYP/6-31G** polar=optrot CPHF=RdFreq'
  echo
  echo $i
  echo
  echo 0 1
  tail xyz/rot$i.xyz -n +3
  echo
  echo 355nm,589.3nm,633nm
  } > gaussian/rot$i.com

  rm rot$i.in

done

