import numpy as np
import torch
from torch_geometric.data import Data
import rdkit
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import ase.io


BOHR_TO_ANG = 0.529177210903

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
}



def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def read_xyz(xyz, bohr=False):
    mol = ase.io.read(xyz)
    if bohr:
        mol.set_positions(mol.positions*ase.units.Bohr)
    return np.array(mol.get_chemical_symbols()), mol.positions


def atom_featurizer(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs(includeNeighbors=True)),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])
    return torch.tensor(atom_features_list)


def get_graph(mol, atomtypes, coords, y, device='cpu'):
    """
    Builds graph object

    data.x -> node features
    data.pos -> xyz coordinates
    data.edge_index -> --
    data.edge_attr -> --
    data.y -> reaction id
    """
    atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
    assert np.all(atoms == atomtypes), "atoms from xyz and smiles don't match"
    assert coords.shape[0] == len(atoms), "different number of atoms"
    assert coords.shape[1] == 3, "wrong dimensionality of coordinates"
    x = atom_featurizer(mol)
    data = Data(x=x, y=torch.tensor(y), pos=torch.tensor(coords, dtype=torch.float32))
    return data.to(device)


def get_empty_graph():
    num_node_feat = atom_featurizer(Chem.MolFromSmiles('C')).shape[-1]
    return Data(x=torch.zeros((0, num_node_feat)), y=torch.tensor(-1), pos=torch.zeros((0,3)))


def sanitize_mol_no_valence_check(mol):
    # rdkit doesn't like "hypervalent" atoms.
    # The standard sanitization would fail even on [SiF6]^{-2}
    # with SMILES 'F[Si-2](F)(F)(F)(F)F' (https://pubchem.ncbi.nlm.nih.gov/compound/Hexafluorosilicate)
    # Solution:
    # https://sourceforge.net/p/rdkit/mailman/message/32599798/
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
