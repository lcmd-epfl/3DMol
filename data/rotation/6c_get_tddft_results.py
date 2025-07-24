import subprocess
import numpy as np
import pandas as pd
from io import StringIO

grepped = subprocess.Popen("grep 'deg\.' gaussian/*.log", shell=True, text=True, stdout=subprocess.PIPE).stdout.read()
vals = np.genfromtxt(StringIO(grepped), usecols=11)
keys = np.genfromtxt(StringIO(grepped), usecols=0, dtype=str)
keys = [*map(lambda x: x.split('.')[0].split('/')[-1], keys)]
computed = {key: val for key, val in zip(keys, vals)}

df = pd.read_csv('data.csv')
df['specific_rotation_computed'] = np.zeros(len(df))
for i in range(len(df)):
    df.loc[i, 'specific_rotation_computed'] = computed[str(df.id[i])]

df.to_csv('data.csv', index=False)
