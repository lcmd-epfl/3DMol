import copy
import numpy as np
import pandas as pd
import ase.io


def signed_volume(A, B, C, D):
    return np.dot(B-A, np.cross(C-A, D-A))


def compute_invariant(mol):
    return signed_volume(mol.positions[0], mol.positions[1], mol.positions[2], mol.positions[3])


def distort_molecule(mol):
    mol = copy.deepcopy(mol)
    mol.euler_rotate(np.random.random()*360, np.random.random()*180, np.random.random()*360)
    mol.set_positions(mol.positions + np.random.rand(*mol.positions.shape)*1e-1)
    return mol


def mirror_molecule(mol):
    mol = copy.deepcopy(mol)
    mol.set_positions(-mol.positions)
    return mol


N = 256
np.random.seed(666)
seed = ase.io.read('seed.xyz')

data = {'idx': [], 'triple': []}

for i in range(N):
    mol = distort_molecule(seed)
    mol.write(f'xyz/{i}.xyz')
    data['idx'].append(str(i))
    data['triple'].append(compute_invariant(mol))

    mol2 = mirror_molecule(mol)
    mol2.write(f'xyz/x{i}.xyz')
    data['idx'].append(f'x{i}')
    data['triple'].append(compute_invariant(mol2))


df = pd.DataFrame(data)

df.to_csv('data.csv', index=False)

