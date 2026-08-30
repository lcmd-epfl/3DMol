#!/usr/bin/env python3

import re
import pandas as pd
from tqdm import trange
import ase.io
from vmol import vmol

symtol = 1e-2
csv_path = 'data_converged.csv'
xyz_dir = 'xyz-xtb'

chiral_group_pattern = re.compile("([CD][0-9]+)$")  # Cn and Dn

df = pd.read_csv(csv_path, dtype={'id': str})
df['point_group'] = None
df['is_point_group_chiral'] = None
df['point_group_noH'] = None
df['is_point_group_noH_chiral'] = None

for i in trange(len(df), disable=False):
    fname = f'{xyz_dir}/{df.loc[i, "id"]}.xyz.xtbopt.xyz'

    pg = vmol.capture(args=[fname, 'gui:0', 'com:.', f'symtol:{symtol}'])
    df.loc[i, 'point_group'] = pg
    df.loc[i, 'is_point_group_chiral'] = bool(chiral_group_pattern.match(pg))

    mol = ase.io.read(fname)
    mol = mol[mol.numbers!=1]
    pg = vmol.capture(mols=mol, args=['gui:0', 'com:.', f'symtol:{symtol}'])
    df.loc[i, 'point_group_noH'] = pg
    df.loc[i, 'is_point_group_noH_chiral'] = bool(chiral_group_pattern.match(pg))

df.to_csv(csv_path, index=False)
