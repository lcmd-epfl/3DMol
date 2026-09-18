#!/usr/bin/env python3

import subprocess
import numpy as np
import pandas as pd
from io import StringIO

df = pd.DataFrame({
    'idx': [f'rot{i:02}' for i in range(24)],
    'point_group': ['C1'] * 24,
    'point_group_chiral': [True] * 24,
    })

for la in ('6330', '5893', '3550'):
    grepped = subprocess.Popen(f"grep '{la}\\.0 A' gaussian/*.log", shell=True, text=True, stdout=subprocess.PIPE).stdout.read()
    vals = np.genfromtxt(StringIO(grepped), usecols=11)
    keys = np.genfromtxt(StringIO(grepped), usecols=0, dtype=str)
    keys = [*map(lambda x: x.split('.')[0].split('/')[-1], keys)]
    computed = {key: val for key, val in zip(keys, vals)}

    df[f'rot{la[:3]}'] = np.zeros(len(df))
    for i in range(len(df)):
        df.loc[i, f'rot{la[:3]}'] = computed[df.idx[i]]

df.loc[0, 'point_group'] = 'Cs'
df.loc[12, 'point_group'] = 'Ci'
df.loc[0, 'point_group_chiral'] = df.loc[12, 'point_group_chiral'] = False

for la in ('633', '589', '355'):
    df[f'rot{la}_sign'] = np.sign(df[f'rot{la}'])

df.to_csv('data.csv', index=False)
