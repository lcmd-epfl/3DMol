import numpy as np
from process.dataloader import MolDataset


class Rotation(MolDataset):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 2  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/rotation/data.csv'
        self.processed_dir='data/rotation/processed/'
        self.smiles_column = None
        self.id_column = 'id'
        self.default_parameters = dict(geometry='xtb', target_column='specific_rotation', graph_method='torchchem_v1')
        bad_indices = None
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/rotation/xyz-xtb/{idx}.xyz.xtbopt.xyz'

        super().__init__(process=process, noH=noH,
                         check=check,
                         bad_indices=bad_indices,
                         verbose=verbose)


class RotationBoth(MolDataset):
    def __init__(self, process=True, verbose=4,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/rotation/data_both.csv'
        self.processed_dir='data/rotation/processed/'
        self.smiles_column = None
        self.id_column = 'id'
        self.default_parameters = dict(geometry='xtb', target_column='specific_rotation', graph_method='torchchem_v1')
        bad_indices = None
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/rotation/xyz-xtb/{idx}.xyz.xtbopt.xyz'

        super().__init__(process=process, noH=noH,
                         check=check,
                         bad_indices=bad_indices,
                         verbose=verbose)
