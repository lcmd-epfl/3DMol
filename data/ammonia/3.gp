set datafile separator ","
plot 'data_single.csv' using 4:5 noti, 2.1026*x+544.84 ti 'R^2=0.35'
set xlabel 'energy, kcal/mol'
set ylabel 'rotation, deg'

