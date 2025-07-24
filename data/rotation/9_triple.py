import numpy as np
import pandas as pd
import ase.io
from tqdm import tqdm


def signed_volume(A, B, C, D):
    return np.dot(B-A, np.cross(C-A, D-A))


def compute_invariant(mol):
    moments, axes = mol.get_moments_of_inertia(vectors=True)
    idx = np.argsort(moments)[::-1]
    moments = moments[idx]
    axes    = axes   [idx]

    r = (mol.get_positions()-mol.get_center_of_mass()) @ axes.T
    #mol.set_positions(r)
    #mol.write('molecule_rotated.xyz')

    ix = np.argsort(abs(r[:,0]))[-1]
    rx = r[ix]
    r = r[np.r_[0:ix,ix+1:len(r)]]

    iy = np.argsort(abs(r[:,1]))[-1]
    ry = r[iy]
    r = r[np.r_[0:iy,iy+1:len(r)]]

    iz = np.argsort(abs(r[:,2]))[-1]
    rz = r[iz]

    return signed_volume(np.array((0,0,0)), rx, ry, rz)


csv = 'data_both.csv'
df = pd.read_csv(csv)
triple_product = np.array([compute_invariant(ase.io.read(f'xyz-xtb/{i}.xyz.xtbopt.xyz')) for i in tqdm(df.id)])
df['triple_product'] = triple_product
assert np.allclose(triple_product.reshape((2,-1))[0], -triple_product.reshape((2,-1))[1], atol=1e-6)
df.to_csv(csv, index=False)
