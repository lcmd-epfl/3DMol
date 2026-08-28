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
import ase.io
from ase.geometry.analysis import Analysis
from process.create_graph import get_graph


class MolDataset(Dataset):

    def print(self, priority, *args):
        if self.verbose>=priority:
            print(*args)

    def __init__(self, process=True, bad_indices=None,
                 embedding_specs=None,
                 extra_args=None,
                 classification=False,
                 noH=True, verbose=1, check=False):

        global_version = 3  # INCREASE WHEN BREAKING CHANGES
        self.noH = noH
        self.graph_method = self.parameters.graph_method
        self.verbose = verbose
        self.check = check
        dataset_prefix = os.path.splitext(os.path.basename(self.csv_path))[0]
        dataset_prefix = f'{dataset_prefix}.{self.parameters.geometry}.{self.parameters.graph_method}'
        if bad_indices:
            dataset_prefix = f'{dataset_prefix}.without_{os.path.splitext(os.path.basename(bad_indices))[0]}'
        if noH:
            dataset_prefix += '.noH'
        self.paths = SimpleNamespace(
                mg=join(self.processed_dir, f'{dataset_prefix}.v{global_version}:{self.version}.mol_graphs.pt'),
                )

        self.print(2, "Loading data into memory...")
        self.print(1, f'{dataset_prefix=}')

        self.df = pd.read_csv(self.csv_path, dtype={self.id_column: str})

        if bad_indices:
            bad_indices = np.loadtxt(bad_indices, dtype=str)
            assert sum(self.df[self.id_column].isin(bad_indices))==len(bad_indices)
            self.df = self.df[~self.df[self.id_column].isin(bad_indices)]
            self.df.reset_index(inplace=True)

        self.nmols = len(self.df)
        self.indices = self.df[self.id_column].to_numpy()
        self.labels = torch.tensor(self.df[self.parameters.target_column].values)
        if self.graph_method in {'smiles', 'smiles_mapped', 'smiles_loose'}:
            self.smiles = self.df[self.smiles_column]

        if process:
            self.print(2, "Processing by request...")
            self.process()
        else:
            if exists(self.paths.mg):
                self.mol_graphs = torch.load(self.paths.mg)
                self.print(2, f"Coords and graphs successfully read from {self.processed_dir}")
            else:
                self.print(2, "Processed data not found, processing data...")
                self.process()

        assert self.nmols==len(self.indices)
        assert self.nmols==len(self.labels)
        assert self.nmols==len(self.mol_graphs)

        if embedding_specs is not None:
            self.extra = [{key: torch.tensor(self.df[key].values[i].item()) for key in embedding_specs.keys()} for i in range(len(self.df))]

        if classification:
            self.check_classes()
        else:
            self.standardize_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        extra = self.extra[idx] if hasattr(self, 'extra') else None
        return self.labels[idx], idx, self.mol_graphs[idx], extra

    def process(self):

        self.print(2, f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            self.print(2, f"Creating processed directory {self.processed_dir}")

        self.mol_graphs = []

        for i, idx in enumerate(tqdm(self.indices, desc="making graphs", disable=not self.verbose)):
            xyz = self.get_xyz_path(idx)
            asemol = self.read_xyz(xyz)

            if self.graph_method in {'smiles', 'smiles_mapped', 'smiles_loose', 'smiles_atoms'}:
                if self.graph_method=='smiles_atoms':
                    smi = '.'.join([f'[{a}:{i+1}]' for i, a in enumerate(asemol.get_chemical_symbols())])
                else:
                    smi = self.smiles[i]
                try:
                    graph = self.make_smiles_graph(smi, asemol, i)
                except Exception as e:
                    print()
                    print(self.indices[i], smi, self.get_xyz_path(self.indices[i]))
                    if not self.check:
                        raise e

            elif self.graph_method == 'torchchem_v1':
                local_mask = self.get_local_mask(asemol=asemol)
                graph = get_graph(None, asemol.numbers, asemol.positions, i, features=self.graph_method, local_mask=local_mask)

            else:
                # other ways to featurize the atoms
                raise NotImplementedError
            self.mol_graphs.append(graph)

        if self.check:
            exit(0)

        torch.save(self.mol_graphs, self.paths.mg)

    def make_smiles_graph(self, smi, asemol, i):
        rdmol = Chem.MolFromSmiles(smi, sanitize=False)
        assert rdmol is not None, f"mol obj {self.indices[i]} is None from smi {smi}"
        self.sanitize_mol_no_valence_check(rdmol)
        atoms, coords = np.array(asemol.get_chemical_symbols()), asemol.positions

        if self.noH:
            rdmol = Chem.RemoveAllHs(rdmol, sanitize=False)
            self.sanitize_mol_no_valence_check(rdmol)

        if self.graph_method in {'smiles', 'smiles_loose'}:
            G2D = self.make_nx_graph_from_rdmol(rdmol)
            G3D = self.make_nx_graph_from_asemol(asemol)
            if self.graph_method=='smiles_loose':
                try:
                    # smiles have more or same bonds than xyz
                    mapping = self.match_graphs(G2D, G3D, i, loose=True)
                except:
                    # smiles have less or same bonds than xyz
                    mapping = self.match_graphs(G3D, G2D, i, loose=True)
                    mapping = {val: key for key, val in mapping.items()}
            elif self.graph_method=='smiles':
                mapping = self.match_graphs(G2D, G3D, i)
            for iat, at in enumerate(rdmol.GetAtoms()):
                at.SetAtomMapNum(int(mapping[iat])+1)

        atom_map = np.array([at.GetAtomMapNum() for at in rdmol.GetAtoms()])
        assert np.all(atom_map>0), f"mol {self.indices[i]} is not atom-mapped"
        assert len(atom_map)==len(atoms), f"mol {self.indices[i]} has a wrong number of atoms"
        atom_map = atom_map.argsort().argsort()  # elements rank
        local_mask = self.get_local_mask(asemol=asemol, rdmol=rdmol)
        return get_graph(rdmol, atoms[atom_map], coords[atom_map], i, features='smiles', local_mask=local_mask)

    def check_classes(self):
        unique_vals = torch.unique(self.labels, sorted=True)
        if len(unique_vals)!=2 or unique_vals[0]!=-1.0 or unique_vals[1]!=1.0:
            raise NotImplementedError(f'Can work only with -1/1 classes ({unique_vals} given)')
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(1.0)

    def standardize_labels(self):
        self.mean = torch.mean(self.labels)
        self.std = torch.std(self.labels)
        self.labels = (self.labels - self.mean)/self.std

    def match_graphs(self, G1, G2, i, loose=False):
        GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
        if loose:
            assert GM.subgraph_is_monomorphic(), f'G2 is not isomorphic to any subgraph of G1 in {i}: {self.smiles[i]}, {self.get_xyz_path(self.indices[i])}'
            return next(GM.subgraph_monomorphisms_iter())
        else:
            assert GM.is_isomorphic(), f"smiles and xyz graphs are not isomorphic in {i}: {self.smiles[i]}, {self.get_xyz_path(self.indices[i])}"
            return next(GM.match())

    def make_nx_graph_from_rdmol(self, mol):
        bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
        atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
        return self.make_nx_graph(atoms, bonds)

    def make_nx_graph_from_asemol(self, mol):
        bonds = np.array([(i,j) for i, js in enumerate(Analysis(mol).unique_bonds[0]) for j in js])
        return self.make_nx_graph(mol.get_chemical_symbols(), bonds)

    def make_nx_graph(self, atoms, bonds):
        G = networkx.Graph()
        G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
        G.add_edges_from(bonds)
        return G

    def read_xyz(self, xyz, bohr=False):
        mol = ase.io.read(xyz)
        if bohr:
            mol.set_positions(mol.positions*ase.units.Bohr)
        if self.noH:
            del mol[mol.numbers == 1]
        return mol

    def sanitize_mol_no_valence_check(self, mol):
        # rdkit doesn't like "hypervalent" atoms.
        # The standard sanitization would fail even on [SiF6]^{-2}
        # with SMILES 'F[Si-2](F)(F)(F)(F)F' (https://pubchem.ncbi.nlm.nih.gov/compound/Hexafluorosilicate)
        # Solution:
        # https://sourceforge.net/p/rdkit/mailman/message/32599798/
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)

    def get_local_mask(self, asemol=None, rdmol=None):
        return None

    def get_parameters(self, vardict):
        return SimpleNamespace(**{key: self.default_parameters[key] if (key not in vardict or vardict[key] is None) else vardict[key]
                                                                    for key in self.default_parameters.keys()})
