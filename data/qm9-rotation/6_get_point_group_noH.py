import re
import subprocess
import pandas as pd
from tqdm import trange
import ase.io
import tempfile

symtol = 1e-2
chiral_group_pattern = re.compile("([CD][0-9]+)$")  # Cn and Dn

df = pd.read_csv('data.csv', dtype={'idx': str})
df['point_group_noH'] = None
df['is_point_group_noH_chiral'] = None

for i in trange(len(df)):
    fname = f'xyz/{df.loc[i, "idx"]}.xyz'

    mol = ase.io.read(fname)
    mol = mol[mol.numbers!=1]

    tmpfile = tempfile.mktemp()+'.xyz'
    ase.io.write(tmpfile, mol)
    pg = subprocess.Popen([f'v {tmpfile} gui:0 symtol:{symtol}'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True, shell=True).communicate(input='.')[0].strip()
    is_chiral = bool(chiral_group_pattern.match(pg))

    df.loc[i, 'point_group_noH'] = pg
    df.loc[i, 'is_point_group_noH_chiral'] = is_chiral

df = df[df['is_point_group_noH_chiral']]
df.drop(labels='is_point_group_noH_chiral', axis=1, inplace=True)
df.to_csv('data.csv', index=False)
