import numpy as np
import pandas as pd

eps = 1.0
df = pd.read_csv('data_raw.csv', dtype={'idx': str})
mask_zero = (df['rot589_abs']<=eps) & (df['rot633_abs']<=eps) & (df['rot355_abs']<=eps)
mask_nosign = (df['rot589']==0) | (df['rot633']==0) | (df['rot355']==0)  # probably chiral but cannot assign class
mask_chiral_and_nonzero  = (~mask_zero) &   df['is_point_group_chiral']
mask_chiral_and_zero     =   mask_zero  &   df['is_point_group_chiral']
mask_achiral_and_nonzero = (~mask_zero) & (~df['is_point_group_chiral'])
mask_achiral_and_zero    =   mask_zero  & (~df['is_point_group_chiral'])

print('good', sum(mask_chiral_and_nonzero))
print('good but remove', sum(mask_achiral_and_zero))
print('weird chiral and zero', sum(mask_chiral_and_zero))
print('weird achiral and nonzero', sum(mask_achiral_and_nonzero))

mask_good = (~mask_nosign) & mask_chiral_and_nonzero
df = df[mask_good]
df.drop(labels='is_point_group_chiral', axis=1, inplace=True)
df.to_csv('data.csv', index=False)
