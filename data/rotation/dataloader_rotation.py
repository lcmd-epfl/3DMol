import numpy as np
from process.dataloader import MolDataset


class Rotation(MolDataset):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 3  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.processed_dir='data/rotation/processed/'
        self.smiles_column = None
        self.id_column = 'id'
        self.default_parameters = dict(geometry='xtb', target_column='specific_rotation', graph_method='torchchem_v1',
                                       _dl_extra_csv_name='data.csv')
        bad_indices = None
        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))
        self.csv_path = f'data/rotation/{self.parameters._dl_extra_csv_name}'  # data.csv or data_both.csv

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/rotation/xyz-xtb/{idx}.xyz.xtbopt.xyz'

        super().__init__(process=process, noH=noH,
                         check=check,
                         bad_indices=bad_indices,
                         verbose=verbose)

