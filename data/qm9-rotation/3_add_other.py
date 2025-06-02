from glob import glob
import os
import pandas as pd
import ase.io
from tqdm import tqdm

df = pd.read_csv('data.csv', dtype={'idx': str})
df = df[df.rot589_sign != 0]

df2 = df.copy(deep=True)

df2.idx = ['x'+str(i) for i in df2.idx]
df2.rot633       *= -1
df2.rot589       *= -1
df2.rot355       *= -1
df2.rot589_sign  *= -1

df3 = pd.concat([df, df2])

df3.to_csv('data_both.csv', index=False)

for i in tqdm(glob('xyz/[0-9]*.xyz')):
    mol = ase.io.read(i)
    mol.set_positions(-mol.positions)
    ase.io.write(f'{os.path.dirname(i)}/x{os.path.basename(i)}', mol)
