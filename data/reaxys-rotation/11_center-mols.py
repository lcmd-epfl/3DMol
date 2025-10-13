#!/usr/bin/env python3

import glob
import ase.io
from tqdm import tqdm

directory = 'xyz-xtb'

for xyz in tqdm(sorted(glob.glob(f'{directory}/*.xyz'))):
    mol = ase.io.read(xyz)
    center = mol.get_positions().mean(axis=0)
    mol.set_positions(mol.get_positions() - center)
    ase.io.write(xyz, mol)
