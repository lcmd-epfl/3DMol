#!/usr/bin/env python3

from glob import glob
import os
import pandas as pd
import ase.io
from tqdm import trange

lambdas = [589, 633, 355]
csv_path = 'data.csv'
csv_both_path = 'data_both.csv'

df = pd.read_csv(csv_path, dtype={'idx': str})
df_mirror = df.copy(deep=True)
df_mirror.idx = ['x'+str(i) for i in df_mirror.idx]
for la in lambdas:
    df_mirror[f'rot{la}_sign']  *= -1
    df_mirror[f'rot{la}']       *= -1
    df_mirror[f'rot{la}_power'] *= -1

df_both = pd.concat([df, df_mirror])
df_both.to_csv(csv_both_path, index=False)

for i in trange(len(df)):
    idx = df.loc[i, "idx"]
    if not os.path.exists(mirror_path := f'xyz/x{idx}.xyz'):
        mol = ase.io.read(f'xyz/{idx}.xyz')
        mol.set_positions(-mol.positions)
        ase.io.write(mirror_path, mol)
