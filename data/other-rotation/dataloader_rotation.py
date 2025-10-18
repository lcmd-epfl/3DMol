import os
import numpy as np
from process.dataloader import MolDataset


class ScanRotation(MolDataset):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 classification=False,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        dirname = os.path.dirname(__file__) + '/scan/'
        self.processed_dir=f'{dirname}/processed/'
        self.smiles_column = None
        self.id_column = 'idx'
        self.default_parameters = dict(geometry='gen', target_column='rot589', graph_method='torchchem_v1',
                                       _dl_extra_csv_name='data.csv',
                                       _dl_extra_bad_idx=None)

        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))
        self.csv_path = f'{dirname}/{self.parameters._dl_extra_csv_name}'   # data.csv or data_both.csv
        bad_indices   = f'{dirname}/{self.parameters._dl_extra_bad_idx}' \
                        if self.parameters._dl_extra_bad_idx else None          # negative_triple_idx.dat

        self.get_xyz_path = lambda idx: f"{dirname}/xyz/{idx.replace('extra_', '')}.xyz"

        super().__init__(process=process, noH=noH,
                         classification=classification,
                         check=check,
                         bad_indices=bad_indices,
                         verbose=verbose)

