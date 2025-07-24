import numpy as np
from process.dataloader import MolDataset


class Yuri(MolDataset):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 classification=False,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 4  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.processed_dir='data/yuri/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='default', target_column='gap', graph_method='torchchem_v1',
                                       _dl_extra_csv_path='data/yuri/refcode_properties_2300.csv')
        bad_indices = None
        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))

        self.csv_path = self.parameters._dl_extra_csv_path

        if self.parameters.geometry=='default':
            self.get_xyz_path = lambda idx: f'data/yuri/0-XYZS/{idx}.xyz'
        elif self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/yuri/1-XYZS_xtb/{idx}_opt.xyz'

        super().__init__(process=process, noH=noH,
                         classification=classification,
                         check=check, bad_indices=bad_indices,
                         verbose=verbose)


    def get_local_mask(self, asemol, **kwargs):
        mask = (asemol.numbers <= 28) * (asemol.numbers >= 24)
        assert sum(mask)==1
        return mask
