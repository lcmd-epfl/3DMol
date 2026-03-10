#!/usr/bin/env python3

import numpy as np
import pandas as pd

path = 'tmPHOTO_property.csv'
n = 5

df = pd.read_csv(path)
df.drop(labels=df.columns.difference(['refcode','gap']), axis=1, inplace=True)

for i in range(n):
    idx = np.loadtxt(f'refcode_subs/tmPHOTO_sub_{i}.txt', dtype=str)
    df1 = df[df['refcode'].isin(idx)].reset_index(drop=True)
    assert len(df1)==len(idx)
    df1.to_csv(f'tmPHOTO_gap_{len(idx)}_{i}.csv', index=False)
