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
df.to_csv(csv_path, index=False)
