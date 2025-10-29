from glob import glob
import numpy as np
import pandas as pd


df = pd.read_csv('data_1_cleaned.csv')
xyzs = [*map(lambda x: int(x.split('/')[1].split('.')[0]), glob('xyz/*.xyz'))]
bad = np.setdiff1d(df.id, xyzs)

df = df[~df['id'].isin(bad)].reset_index(drop=True)
df.to_csv('data_2_cleaned.csv', index=False)
