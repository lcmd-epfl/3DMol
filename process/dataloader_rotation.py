import numpy as np
from process.dataloader import MolDataset


class Rotation(MolDataset):
    def __init__(self, process=True, verbose=4, target_column='specific_rotation',
                 xtb=True,
                 noH=True, graph_method='torchchem_v1', check=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/rotation/data.csv'
        self.processed_dir='data/rotation/processed/'
        self.smiles_column = None
        self.id_column = 'id'
        bad_indices = None
        if xtb is False:
            raise RuntimeError
        self.get_xyz_path = lambda idx: f'data/rotation/xyz-xtb/{idx}.xyz.xtbopt.xyz'
        geometry = 'xtb'

        super().__init__(process=process, geometry=geometry, noH=noH,
                         target_column=target_column, check=check,
                         bad_indices=bad_indices,
                         graph_method=graph_method, verbose=verbose)
