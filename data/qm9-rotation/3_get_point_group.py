import re
import subprocess
import pandas as pd
from tqdm import trange

symtol = 1e-2
chiral_group_pattern = re.compile("([CD][0-9]+)$")  # Cn and Dn

df = pd.read_csv('data_raw.csv', dtype={'idx': str})
df['point_group'] = None
df['is_point_group_chiral'] = None

for i in trange(len(df)):
    fname = f'xyz/{df.loc[i, "idx"]}.xyz'
    pg = subprocess.Popen([f'v {fname} gui:0 symtol:{symtol}'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True, shell=True).communicate(input='.')[0].strip()
    df.loc[i, 'point_group'] = pg
    df.loc[i, 'is_point_group_chiral'] = bool(chiral_group_pattern.match(pg))

df.to_csv('data_raw.csv', index=False)
