#!/usr/bin/env python3

import os
from collections import Counter
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from tqdm import tqdm
import ase.data


url = 'https://zenodo.org/records/13380412/files/qm9-or.npy'
elements = np.array(['H', 'C', 'N', 'O', 'F'])
lambdas = [633, 589, 355]  # order from Author's github https://github.com/bcmort/OHECC and verified with Gaussian
max_idx_len = 6  # indices are 6-number strings

orig_path = 'qm9-or.npy'
csv_path = 'data_raw.csv'
xyz_dir = 'xyz'
overwrite_xyz = False


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


def extract_xyz(idx, xyz):
    fname = f'{xyz_dir}/{idx}.xyz'
    atoms = xyz[:,3:]
    atoms = elements[np.where(atoms==1.0)[1]]
    if overwrite_xyz or not os.path.isfile(fname):
        coords = xyz[:,:3]
        coords = coords[:len(atoms)]
        coords -= coords.mean(axis=0)
        write_xyz(fname, idx, atoms, coords)
    return atoms


def get_mass(atoms):
    # looks like g16 uses non common masses but these or similar
    c = Counter(atoms)
    return sum(ase.data.atomic_masses[ase.data.atomic_numbers[sym]] * num for sym, num in c.items())


def main():
    if not os.path.isfile(orig_path):
        download(orig_path, url)

    data = np.load(orig_path, allow_pickle=True)
    idx = np.zeros(len(data), dtype=f'<U{max_idx_len}')
    rot = np.zeros((len(data), len(lambdas)))
    mass = np.zeros(len(data))

    for i, entry in enumerate(tqdm(data)):
        idx[i] = entry['index']
        rot[i,:] = entry['rotation']
        atoms = extract_xyz(idx[i], entry['xyz'])
        mass[i] = get_mass(atoms)

    df = pd.DataFrame({'idx': idx, 'mass': mass} | {f'rot{la}': rot[:,i] for i, la in enumerate(lambdas)})
    df.to_csv(csv_path, index=False)


if __name__=='__main__':
    main()
