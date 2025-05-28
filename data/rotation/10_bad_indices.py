import numpy as np
import pandas as pd

csv = 'data_both.csv'
df = pd.read_csv(csv)

negative_triple = df[df.triple_product<0].id.to_numpy()
assert len(negative_triple)==len(df)//2

np.savetxt('negative_triple_idx.dat', negative_triple, fmt='%s')
