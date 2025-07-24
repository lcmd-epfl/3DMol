set datafile separator ","
binstart=0
binwidth=16
plot 'data_single.csv' using (binwidth*(floor(($4-binstart)/binwidth)+0.5)+binstart):(1.0) smooth frequency with boxes fillstyle solid noti
set xlabel 'relative energy, kcal/mol'
set ylabel 'counts'
