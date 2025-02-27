from glob import glob
import os
import pandas as pd
import ase.io

df = pd.read_csv('data.csv')
df2 = df.copy(deep=True)

df2.id = ['x'+str(i) for i in df2.id]
df2.specific_rotation *= -1
df2.specific_rotation_computed *= -1
df2.weights_signed *= -1
df2.drop(['SMILES', 'canon_SMILES'], axis=1, inplace=True)

df3 = pd.concat([df, df2])
df3.to_csv('data_both.csv', index=False)


for i in glob('xyz-xtb/[0-9]*.xyz'):
    mol = ase.io.read(i)
    mol.set_positions(-mol.positions)
    ase.io.write(f'{os.path.dirname(i)}/x{os.path.basename(i)}', mol)
