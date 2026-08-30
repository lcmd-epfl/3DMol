import os
from process.dataloader import MolDataset


class RXRotation(MolDataset):
    def __init__(self, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        cwd = os.path.dirname(__file__)

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.processed_dir=f'{cwd}/processed/'
        self.smiles_column = None
        self.id_column = 'idx'
        self.default_parameters = dict(geometry='rdkit', target_column='rot_sign', graph_method='torchchem_v1',
                                       _dl_extra_csv_name='data.csv',
                                       _dl_extra_bad_idx=None)

        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))
        self.csv_path = f'{cwd}/{self.parameters._dl_extra_csv_name}'   # data.csv or data_both.csv

        self.get_xyz_path = lambda idx: f'{cwd}/xyz/{idx}.xyz' if type(idx)==str else f'data/qm9-rotation/xyz/{idx:06d}.xyz'

        super().__init__(verbose=verbose, check=check, **kwargs)

