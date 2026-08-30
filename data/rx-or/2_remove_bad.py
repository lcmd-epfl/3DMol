#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from tqdm import trange

csv_path = 'data_raw.csv'
csv_clean_path = 'data_no_bad.csv'

df = pd.read_csv(csv_path, dtype={'idx': str})

aggregate_mask = df['nmol']>1
bad_mask = aggregate_mask | df['R=0'] | df['close']

print('more than 1 mol: ', sum(aggregate_mask))
print('no coordinates: ', sum(df['R=0']))
print('bad coordinates: ', sum(df['close']))
print('total bad: ', sum(bad_mask))

df = df[~bad_mask].reset_index(drop=True)
df.drop(['nmol', 'R=0', 'close'], axis=1, inplace=True)
df.to_csv(csv_clean_path, index=False)
