#!/usr/bin/env python3

import re
import pandas as pd
from tqdm import trange
import ase.io
from vmol import vmol

symtol = 1e-2
csv_path = 'data.csv'
xyz_dir = 'xyz'

chiral_group_pattern = re.compile("([CD][0-9]+)$")  # Cn and Dn

df = pd.read_csv(csv_path, dtype={'idx': str})
df['point_group_noH'] = None
df['is_point_group_noH_chiral'] = None

for i in trange(len(df)):
    mol = ase.io.read(f'{xyz_dir}/{df.loc[i, "idx"]}.xyz')
    mol = mol[mol.numbers!=1]
    pg = vmol.capture(mols=mol, args=['gui:0', 'com:.', f'symtol:{symtol}'])
    is_chiral = bool(chiral_group_pattern.match(pg))
    df.loc[i, 'point_group_noH'] = pg
    df.loc[i, 'is_point_group_noH_chiral'] = is_chiral

print('*** {} total, {} ({:.0f}%) kept, {} ({:.0f}%) removed'.format(tot:=len(df), kept:=sum(df['is_point_group_noH_chiral']), kept/tot*100, removed:=tot-kept, removed/tot*100))
df = df[df['is_point_group_noH_chiral']]
df.drop(labels='is_point_group_noH_chiral', axis=1, inplace=True)
df.to_csv(csv_path, index=False)
