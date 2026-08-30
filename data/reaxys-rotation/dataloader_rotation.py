import os
import numpy as np
from process.dataloader import MolDataset


class Rotation(MolDataset):
    def __init__(self, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        self.version = 4  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        dirname = os.path.dirname(__file__)
        geometry = 'xtb'
        self.processed_dir=f'{dirname}/processed/'
        self.smiles_column = None
        self.id_column = 'id'
        self.default_parameters = dict(geometry='xtb', target_column='specific_rotation', graph_method='torchchem_v1',
                                       _dl_extra_csv_name='data.csv',
                                       _dl_extra_bad_idx=None)

        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))
        self.csv_path = f'{dirname}/{self.parameters._dl_extra_csv_name}'   # data.csv or data_both.csv
        bad_indices   = f'{dirname}/{self.parameters._dl_extra_bad_idx}' \
                        if self.parameters._dl_extra_bad_idx else None          # negative_triple_idx.dat

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'{dirname}/xyz-xtb/{idx}.xyz.xtbopt.xyz'

        super().__init__(bad_indices=bad_indices, verbose=verbose, check=check, **kwargs)

