#!/usr/bin/env python3

import numpy as np
import pandas as pd

csv_path = 'data_raw.csv'
lambdas = [589, 633, 355]

df = pd.read_csv(csv_path, dtype={'idx': str})

for la in lambdas:
    rot = df[f'rot{la}'].to_numpy()
    df[f'rot{la}_sign'] = np.sign(rot)
    df[f'rot{la}_abs']  = np.abs(rot)
    power = rot * df['mass'].to_numpy() / 100.0
    df[f'rot{la}_power'] = power
    df[f'rot{la}_power_abs'] = np.abs(power)

df.drop('mass', axis=1, inplace=True)
df.to_csv(csv_path, index=False)
