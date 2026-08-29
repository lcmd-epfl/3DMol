import numpy as np
import torch
from torch_geometric import data as tgdata
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from process.feature import atom_geom

try:
    torch.serialization.add_safe_globals([tgdata.data.DataEdgeAttr, tgdata.data.DataTensorAttr, tgdata.storage.GlobalStorage])
except:
    pass


def get_graph(mol, atoms, coords, y, features, device='cpu', local_mask=None):
    """
    Builds graph object

    data.x -> node features
    data.pos -> xyz coordinates
    data.edge_index -> --
    data.edge_attr -> --
    data.y -> reaction id
    """
    assert coords.shape[0] == len(atoms), "different number of atoms"
    assert coords.shape[1] == 3, "wrong dimensionality of coordinates"
    if features=='smiles':
        atoms1 = np.array([at.GetSymbol() for at in mol.GetAtoms()])
        assert np.all(atoms1 == atoms), "atoms from xyz and smiles don't match"
        x = smiles_featurizer(mol)
    elif features=='torchchem_v1':
        x = atom_geom(atoms, torch.tensor(coords))
        x = x.type(torch.Tensor)

    else:
        raise NotImplementedError
    if local_mask is None:
        data = tgdata.Data(x=x, y=torch.tensor(y), pos=torch.tensor(coords, dtype=torch.float32))
    else:
        data = tgdata.Data(x=x, y=torch.tensor(y), pos=torch.tensor(coords, dtype=torch.float32),
                    local_mask=torch.tensor(local_mask, dtype=torch.bool))
    return data.to(device)


def get_empty_graph(features='smiles'):
    if features=='smiles':
        num_node_feat = smiles_featurizer(Chem.MolFromSmiles('C')).shape[-1]
    else:
        raise NotImplementedError
    return tgdata.Data(x=torch.zeros((0, num_node_feat)), y=torch.tensor(-1), pos=torch.zeros((0,3)))


def smiles_featurizer(mol):
    allowable_features = {
        'possible_atomic_num_list': [*list(range(1, 119)), 'misc'],
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
