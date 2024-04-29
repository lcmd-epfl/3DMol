import os
from os.path import exists, join
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import networkx
import networkx.algorithms.isomorphism as iso
from process.create_graph import get_graph, read_xyz, sanitize_mol_no_valence_check


class MolDataset(Dataset):

    def print(self, priority, *args):
        if self.verbose>=priority:
            print(*args)


    def __init__(self, process=True, geometry='dft', noH=True, atom_mapping=False, verbose=1):
        self.noH = noH
        self.verbose = verbose
        dataset_prefix = os.path.splitext(os.path.basename(self.csv_path))[0]
        dataset_prefix = f'{dataset_prefix}.{geometry}'
        if noH:
            dataset_prefix += '.noH'
        self.paths = SimpleNamespace(
                mg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.mol_graphs.pt'),
                )

        self.print(2, "Loading data into memory...")
        self.print(1, f'{dataset_prefix=}')

        self.df = pd.read_csv(self.csv_path)
        self.nmols = len(self.df)
        self.indices = self.df[self.id_column].to_list()
        self.labels = torch.tensor(self.df[self.target_column].values)
        self.smiles = self.df[self.smiles_column]

        if process == True:
            self.print(2, "Processing by request...")
            self.process()
        else:
            if exists(self.paths.mg):
                self.mol_graphs = torch.load(self.paths.mg)
                self.print(2, f"Coords and graphs successfully read from {self.processed_dir}")
            else:
                self.print(2, "Processed data not found, processing data...")
                self.process()

        self.standardize_labels()


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        mol = self.mol_graphs[idx]
        label = self.labels[idx]
        return self.labels[idx], idx, self.mol_graphs[idx]


    def process(self):

        self.print(2, f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            self.print(2, f"Creating processed directory {self.processed_dir}")

        self.mol_graphs = []

        for i, idx in (enumerate(tqdm(self.indices, desc="making graphs")) if self.verbose>=1 else self.indices):
            xyz = self.get_xyz_path(idx)
            atomtypes, coords = read_xyz(xyz)
            smi = self.smiles[i]
            graph = self.make_graph(smi, atomtypes, coords,  f'r{idx}', i, None)
            self.mol_graphs.append(graph)

        torch.save(self.mol_graphs, self.paths.mg)


    def make_graph(self, smi, atoms, coords, ireact, idx, smi2=None):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        assert mol is not None, f"mol obj {ireact} is None from smi {smi}"
        sanitize_mol_no_valence_check(mol)

        if self.noH:
            mol = Chem.RemoveAllHs(mol, sanitize=False)
            sanitize_mol_no_valence_check(mol)
            noH_idx = np.where(atoms!='H')
            atoms = atoms[noH_idx]
            coords = coords[noH_idx]

        atom_map = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
        assert np.all(atom_map>0), f"mol {ireact} is not atom-mapped"
        assert len(atom_map)==len(atoms), f"mol {ireact} has a wrong number of atoms"
        atom_map = atom_map.argsort().argsort()  # elements rank

        return get_graph(mol, atoms[atom_map], coords[atom_map], idx)


    def standardize_labels(self):
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std


    def make_nx_graph_from_mol(self, mol):
        bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
        atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
        G = networkx.Graph()
        G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
        G.add_edges_from(bonds)
        return G


class PropargReactants(MolDataset):
    def __init__(self, process=True, xtb=False, noH=True, atom_mapping=False,
                 verbose=4):

        self.version = 0.1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/proparg/data_reactants.csv'
        self.processed_dir='data/proparg/processed/'
        self.smiles_column = 'smiles_mapped'
        self.id_column = 'xyz_id'
        self.target_column = 'Eafw'
        if xtb:
            files_dir='data/proparg/xyz-xtb/'
            geometry = 'xtb'
        else:
            files_dir='data/proparg/xyz/'
            geometry = 'dft'
        self.get_xyz_path = lambda idx: f'{files_dir}/{idx}.r.xyz'

        super().__init__(process=process, geometry=geometry, noH=noH, atom_mapping=atom_mapping, verbose=verbose)
