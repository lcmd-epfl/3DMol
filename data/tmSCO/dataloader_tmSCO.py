import numpy as np
from process.dataloader import MolDataset


class tmSCO(MolDataset):
    def __init__(self, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        self.version = 2  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        wd = 'os.path.dirname(__file__)'
        self.csv_path=f'{wd}/tmSCO_property.csv'
        self.processed_dir=f'{wd}/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='xtb', target_column='gap', graph_method='torchchem_v1')
        bad_indices = None
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'{wd}/xyz/{idx}.xyz'

        super().__init__(bad_indices=bad_indices, verbose=verbose, check=check, **kwargs)


    def get_local_mask(self, asemol, **kwargs):
        metals = np.array(['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Mo', 'Ru'])[:,None]
        mask = (np.array(asemol.symbols)==metals).sum(axis=0)
        assert sum(mask)==1
        assert len(mask)==len(asemol)
        return mask
