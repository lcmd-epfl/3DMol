#!/usr/bin/env python3

import numpy as np
import pandas as pd

eps = 1.0
csv_raw_path = 'data_raw.csv'
csv_clean_path = 'data.csv'
lambdas = [589, 633, 355]

df = pd.read_csv(csv_raw_path, dtype={'idx': str})

mask_zero   = (df[ [f'rot{la}_abs'  for la in lambdas] ] <= eps).all(axis=1)
mask_nosign = (df[ [f'rot{la}_sign' for la in lambdas] ] == 0).any(axis=1)

mask_chiral_and_nonzero  = (~mask_zero) &   df['is_point_group_chiral']
mask_chiral_and_zero     =   mask_zero  &   df['is_point_group_chiral']
mask_achiral_and_nonzero = (~mask_zero) & (~df['is_point_group_chiral'])
mask_achiral_and_zero    =   mask_zero  & (~df['is_point_group_chiral'])
mask_good = (~mask_nosign) & mask_chiral_and_nonzero

print(f'chiral and at least one |α|>{eps}:', sum(mask_chiral_and_nonzero))
print(f'achiral and all |α|≤{eps}:', sum(mask_achiral_and_zero))
print(f'weird: chiral and all |α|≤{eps}:', sum(mask_chiral_and_zero))
print(f'weird: achiral and at least one |α|>{eps}:', sum(mask_achiral_and_nonzero))
print(f'weird: chiral and at least one |α|>{eps} but at least one |α|=0:', sum(mask_nosign & mask_chiral_and_nonzero))
print('*** {} total, {} ({:.0f}%) kept, {} ({:.0f}%) removed'.format(tot:=len(df), kept:=sum(mask_good), kept/tot*100, removed:=tot-kept, removed/tot*100))

df = df[mask_good]
df.drop(labels='is_point_group_chiral', axis=1, inplace=True)
df.to_csv(csv_clean_path, index=False)
