import numpy as np
import pandas as pd

df = pd.read_csv('data_raw.csv', dtype={'idx': str})
for la in (589, 633, 355):
    rot = df[f'rot{la}'].to_numpy()
    df[f'rot{la}_sign'] = np.sign(rot)
    df[f'rot{la}_abs']  = np.abs(rot)
df.to_csv('data_raw.csv', index=False)
