#on lc3

module load gaussian/g16/C.01

for i in gaussian/*.com; do
  echo $i;
  g16 $i ;
done
