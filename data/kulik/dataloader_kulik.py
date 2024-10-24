import numpy as np
from process.dataloader import MolDataset


class tmPHOTO(MolDataset):
    def __init__(self, process=True, verbose=4,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/kulik/tmPHOTO_property.csv'
        self.processed_dir='data/kulik/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='xtb', target_column='gap', graph_method='torchchem_v1')
        bad_indices = None
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/kulik/tmPHOTO-xyz/{idx}.xyz'

        super().__init__(process=process, noH=noH,
                         check=check, bad_indices=bad_indices,
                         verbose=verbose)


    def get_local_mask(self, asemol, **kwargs):
        metals = np.array(['Fe', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ag', 'Cd', 'Re', 'Ir', 'Pt', 'Au', 'Hg'])[:,None]
        mask = (np.array(asemol.symbols)==metals).sum(axis=0)
        assert sum(mask)==1
        assert len(mask)==len(asemol)
        return mask


class tmSCO(MolDataset):
    def __init__(self, process=True, verbose=4,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/kulik/tmSCO_property.csv'
        self.processed_dir='data/kulik/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='xtb', target_column='gap', graph_method='torchchem_v1')
        bad_indices = None
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/kulik/tmSCO-xyz/{idx}.xyz'

        super().__init__(process=process, noH=noH,
                         check=check, bad_indices=bad_indices,
                         verbose=verbose)


    def get_local_mask(self, asemol, **kwargs):
        metals = np.array(['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Mo', 'Ru'])[:,None]
        mask = (np.array(asemol.symbols)==metals).sum(axis=0)
        assert sum(mask)==1
        assert len(mask)==len(asemol)
        return mask
