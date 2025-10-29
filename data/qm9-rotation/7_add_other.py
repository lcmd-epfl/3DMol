from glob import glob
import os
import pandas as pd
import ase.io
from tqdm import trange

df = pd.read_csv('data.csv', dtype={'idx': str})
df2 = df.copy(deep=True)
df2.idx = ['x'+str(i) for i in df2.idx]
for la in (589, 633, 355):
    df2[f'rot{la}_sign'] *= -1
    df2[f'rot{la}']      *= -1

df3 = pd.concat([df, df2])
df3.to_csv('data_both.csv', index=False)

for i in trange(len(df)):
    idx = df.loc[i, "idx"]
    if not os.path.exists(f'xyz/x{idx}.xyz'):
        mol = ase.io.read(f'xyz/{idx}.xyz')
        mol.set_positions(-mol.positions)
        ase.io.write(f'xyz/x{idx}.xyz', mol)
