#!/usr/bin/env python3

import re
import pandas as pd
from tqdm import trange
from vmol import vmol

symtol = 1e-2
csv_path = 'data_raw.csv'
xyz_dir = 'xyz'

chiral_group_pattern = re.compile("([CD][0-9]+)$")  # Cn and Dn

df = pd.read_csv(csv_path, dtype={'idx': str})
df['point_group'] = None
df['is_point_group_chiral'] = None

for i in trange(len(df)):
    pg = vmol.capture(args=[f'{xyz_dir}/{df.loc[i, "idx"]}.xyz', 'gui:0', 'com:.', f'symtol:{symtol}'])
    df.loc[i, 'point_group'] = pg
    df.loc[i, 'is_point_group_chiral'] = bool(chiral_group_pattern.match(pg))

df.to_csv(csv_path, index=False)
