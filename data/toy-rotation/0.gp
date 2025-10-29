set datafile separator ","
binstart=0
binwidth=0.01
plot 'data_single.csv' using (binwidth*(floor(($2-binstart)/binwidth)+0.5)+binstart):(1.0) smooth frequency with boxes fillstyle solid noti
set xlabel 'triple product'
set ylabel 'counts'
