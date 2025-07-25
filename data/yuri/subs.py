#!/usr/bin/env python3

import numpy as np
import pandas as pd

path = 'Property_TM-GSspinPlus_2260.csv'
n = 5

df = pd.read_csv(path)
df.drop(labels=df.columns.difference(['refcode','gap']), axis=1, inplace=True)

for i in range(n):
    idx = np.loadtxt(f'refcode_subs/total_sub_{i}.txt', dtype=str)
    df1 = df[df['refcode'].isin(idx)].reset_index(drop=True)
    assert len(df1)==len(idx)
    df1.to_csv(f'gap_TM-GSspinPlus_{len(idx)}_{i}.csv', index=False)
