import numpy as np
import pandas as pd

df = pd.read_csv('data.csv', dtype={'idx': str})
df['rot589_sign'] = np.sign(df.rot589.to_numpy())
df.to_csv('data.csv', index=False)
