#!/usr/bin/env python3

import os
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from rdkit import Chem, RDLogger

url = 'https://ndownloader.figshare.com/files/66386654' # doi.org/10.6084/m9.figshare.32923205
max_idx_len = 6  # indices are 6-number strings
orig_path = 'rx-or.pickle'
csv_path = 'data_raw.csv'
xyz_dir = 'xyz'
overwrite_xyz = False
RDLogger.DisableLog('rdApp.*')


def download(fpath, url):
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=fpath, reporthook=t.update_to, data=None)
        t.total = t.n


def write_xyz(fpath, comment, atoms, coords):
    with open(fpath, 'w') as f:
        print(len(atoms), file=f)
        print(comment, file=f)
        for q, r in zip(atoms, coords):
            print(q, *r, file=f)


def extract_xyz(row):
    # xyz are already centered
    idx = row['idx']
    fname = f'{xyz_dir}/{idx}.xyz'
    if overwrite_xyz or not os.path.isfile(fname):
        mol = Chem.MolFromSmiles(row['SMILES'])
        mol = Chem.AddHs(mol)
        q = [a.GetSymbol() for a in mol.GetAtoms()]
        write_xyz(fname, idx, q, row['xyz'])


def main():
    if not os.path.isfile(orig_path):
        download(orig_path, url)

    df = pd.read_pickle(orig_path)
    df['idx'] = df['Unnamed: 0'].map(lambda x: f'{x:0{max_idx_len}}')
    df['rot_sign'] = df['Rotation'].map(lambda x: 1 if x=='+' else -1)
    df['R=0'] = False
    for i, row in tqdm(df.iterrows(), total=len(df), disable=False):
        df.iloc[i, df.columns.get_loc('R=0')] = np.all(row['xyz']==0)
        extract_xyz(row)

    df = df[['idx', 'rot_sign', 'SMILES', 'R=0']]
    df.to_csv(csv_path, index=False)


if __name__=='__main__':
    main()
