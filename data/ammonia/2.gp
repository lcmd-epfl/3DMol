set datafile separator ","
binstart=0
binwidth=32
plot 'data_single.csv' using (binwidth*(floor(($5-binstart)/binwidth)+0.5)+binstart):(1.0) smooth frequency with boxes fillstyle solid noti
set xlabel 'rotation, deg'
set ylabel 'counts'
