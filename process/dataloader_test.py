import io
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from process.create_graph import get_graph


class TestSet(Dataset):
    def __init__(self):
        self.mol_graphs = []
        self.process_homometric()
        self.process_chiral()
        self.nmols = len(self.mol_graphs)
        self.labels = torch.arange(self.nmols)
        self.indices = [*range(self.nmols)]
        self.std = torch.tensor(1.0)


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        mol = self.mol_graphs[idx]
        label = self.labels[idx]
        return self.labels[idx], idx, self.mol_graphs[idx]


    def process_homometric(self):
        atoms  = np.array(['He', 'He', 'He', 'He'])
        coords1 = np.array([[0,  0, 0,], [1,  0, 0], [0,  2, 0], [1,  2, 0]])
        coords2 = np.array([[0,  0, 0,], [1,  0, 0], [-1,  0, 0], [0,  2, 0]])
        mol = Chem.MolFromSmiles('.'.join([f'[{a}]' for a in atoms]))
        self.mol_graphs.append(get_graph(mol, atoms, coords1, 0, features='smiles'))
        self.mol_graphs.append(get_graph(mol, atoms, coords2, 1, features='smiles'))


    def process_chiral(self):
        atoms = ['C', 'H', 'F', 'Cl', 'Br']
        mol = Chem.MolFromSmiles('.'.join([f'[{a}]' for a in atoms]))
        f_handler = io.StringIO(""" 0.05928331  -0.12694591  -0.06683713
 0.10695432  -0.39791163  -1.12442036
 0.21441193  -1.20602809   0.72154781
 1.36247485   1.07091085   0.25382473
-1.74312440   0.65997478   0.21588495""")
        coords1 = np.loadtxt(f_handler)
        coords2 = -coords1
        f_handler.close()
        self.mol_graphs.append(get_graph(mol, atoms, coords1, 2, features='smiles'))
        self.mol_graphs.append(get_graph(mol, atoms, coords2, 3, features='smiles'))
