from glob import glob
import os
import pandas as pd
import ase.io

df = pd.read_csv('data.csv')

df2 = pd.DataFrame({
    'id'                         : ['x'+str(i) for i in df.id.to_list()],
    'specific_rotation'          : -df.specific_rotation,
    'specific_rotation_computed' : -df.specific_rotation_computed,
    'weights'                    : df.weights,
    'weights_signed'             : -df.weights_signed,
    })

df3 = pd.concat([df, df2])
df3.to_csv('data_both.csv', index=False)


for i in glob('xyz-xtb/[0-9]*.xyz'):
    mol = ase.io.read(i)
    mol.set_positions(-mol.positions)
    ase.io.write(f'{os.path.dirname(i)}/x{os.path.basename(i)}', mol)
