#!/usr/bin/env python3

import re
import pandas as pd
import ase.io
from tqdm import trange
from vmol import vmol

csv_path = 'data_raw.csv'
xyz_dir = 'xyz'

chiral_group_pattern = re.compile("([CD][0-9]+)$")  # Cn and Dn

def get_pg(mol, symtol=1e-2):
    pg = vmol.capture(mols=mol, args=['gui:0', 'com:.', f'symtol:{symtol}'])
    return pg, bool(chiral_group_pattern.match(pg))


df = pd.read_csv(csv_path, dtype={'idx': str})
df['point_group'] = None
df['is_point_group_chiral'] = None
df['point_group_noH'] = None
df['is_point_group_noH_chiral'] = None

for i in trange(len(df)):
    mol = ase.io.read(f'{xyz_dir}/{df.loc[i, "idx"]}.xyz')
    df.loc[i, 'point_group'], df.loc[i, 'is_point_group_chiral'] = get_pg(mol)
    mol = mol[mol.numbers!=1]
    df.loc[i, 'point_group_noH'] , df.loc[i, 'is_point_group_noH_chiral'] = get_pg(mol)

df.to_csv(csv_path, index=False)
