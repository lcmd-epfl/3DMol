#!/usr/bin/env python3

import os
import pandas as pd
from tqdm import trange

csv_path = 'data_no_bad.csv'
csv_clean_path = 'data.csv'

df = pd.read_csv(csv_path, dtype={'idx': str})
bad_mask = ~(df['is_point_group_chiral'] & df['is_point_group_noH_chiral'])

print('total mols:   ', len(df))
print('achiral:      ', sum(~df['is_point_group_chiral']))
print('achiral w/o H:', sum(~df['is_point_group_noH_chiral']))
print('total bad:    ', sum(bad_mask))

df = df[~bad_mask].reset_index(drop=True)
df.drop(['is_point_group_chiral', 'is_point_group_noH_chiral'], axis=1, inplace=True)
df.to_csv(csv_clean_path, index=False)
