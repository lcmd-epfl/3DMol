import numpy as np
import pandas as pd
import ase.io
from tqdm import tqdm

csv = 'data_both.csv'
df = pd.read_csv(csv)
df['if_specific_rotation_positive'] = np.sign(df.specific_rotation)
df.to_csv(csv, index=False)
