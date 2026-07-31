#!/usr/bin/env python3

import numpy as np
import pandas as pd

for path, col in [
        ('data_raw.csv', 'point_group'),
        ('data.csv', 'point_group'),
        ('data.csv', 'point_group_noH'),
         ]:
    print(f'{path=} {col=}')
    df = pd.read_csv(path, dtype={'idx': str})
    pg = df[col].to_numpy()
    for i in sorted(zip(*np.unique(pg, return_counts=True)), key=lambda x: -x[1]):
        print('{}\t{}'.format(*i))
    print()
