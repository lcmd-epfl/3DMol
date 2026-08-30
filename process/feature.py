# https://github.com/pregHosh/torchchem/blob/12e5f03e9f1a2cf2b24c3fdc6478720c3dca112b/torchdrug/data/feature.py

import warnings
import torch
import networkx as nx
import numpy as np
import scipy.spatial
from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii
from cell2mol.elementdata import ElementData
from morfeus import SASA
# from libarvo import atomic_vs


def get_radii(z, radii="covalent"):
    if radii == "covalent":
        return np.array([covalent_radii[i] for i in z], dtype=float)
    if radii == "vdw":
        return np.array([vdw_radii[i] for i in z], dtype=float)


def get_other_features(z):
    ed = ElementData()
    sym = [ed.elementsym[zi] for zi in z]
    ve = [ed.valenceelectrons[symi] for symi in sym]
    en = [ed.ElectroNegativityPauling[symi] for symi in sym]
    return ve, en


def get_dm(coordinates):
    return scipy.spatial.distance.pdist(coordinates.numpy())


def get_am(z, radii, dm, scale_factor=1.15):
    n = len(z)
    am = np.zeros((n, n), dtype=float)
    row, col = np.triu_indices(n, 1)
    rm = scale_factor * scipy.spatial.distance.pdist(
        radii.reshape(-1, 1), metric=lambda x, y: (x + y).sum()
    )
    am[row, col] = am[col, row] = dm - rm
    return am < 0


def dict2array(d):
    # just in case do not assume the insertion order is the same as atom order
    a = np.array(list(d.items())).T
    return a[1,a[0].argsort()]


def atom_geom(z, coords, label):
    """
    Compute the geometric features of the atoms in the molecule.
    params:
    z: np.array of atomic numbers
    coords: torch.tensor of atomic coordinates

    returns:
    geom_node_feat: torch.tensor of geometric features of the atoms in the molecule
    - degree: Total degree of the atom
    - a_volume: atomic volume
    - a_surface: atomic surface
    - occ_v: occupied volume
    - ve: valence electrons
    - en: pauling electronegativity

    """
    TOL = 0.2
    cov_radii = get_radii(z, radii="covalent")
    vdw_radii = get_radii(z, radii="vdw")
    cov_dm = get_dm(coords)
    cov_am = get_am(z, cov_radii, cov_dm, scale_factor=1.15)
    G = nx.from_numpy_array(cov_am, create_using=nx.Graph)
    degree = [val for (i, val) in G.degree()]
    if not (cov_dm > TOL).all():
        warnings.warn(f"Some atoms are incredibly close to each other! {label}", stacklevel=2)

    # a_volume, a_surface = atomic_vs(coords, vdw_radii)
    sasa = SASA(z, coords.numpy(), vdw_radii, probe_radius=0.0, density=0.01)
    sa_surface = dict2array(sasa.atom_areas)
    sa_volume = dict2array(sasa.atom_volumes)

    ve, en = get_other_features(z)
    ref_v = np.pi * (4 / 3) * vdw_radii**3
    occ_v = (
        ref_v - sa_volume
    ) / ref_v

    geom_node_feat = torch.tensor(np.array(
        [
            degree,
            sa_volume / 10,
            sa_surface / 10,
            occ_v / 10,
            ve,
            en,
        ]
    ).T)
    return geom_node_feat
