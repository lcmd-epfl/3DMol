import numpy as np
from process.dataloader import MolDataset


class tmPHOTO(MolDataset):
    def __init__(self, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        self.version = 3  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        wd = 'os.path.dirname(__file__)'
        self.processed_dir=f'{wd}/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='xtb', target_column='gap', graph_method='torchchem_v1',
                                       _dl_extra_csv_path=f'{wd}/tmPHOTO_property.csv')
        bad_indices = None
        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))

        self.csv_path = self.parameters._dl_extra_csv_path

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'{wd}/xyz/{idx}.xyz'

        super().__init__(bad_indices=bad_indices, verbose=verbose, check=check, **kwargs)


    def get_local_mask(self, asemol, **kwargs):
        metals = np.array(['Fe', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ag', 'Cd', 'Re', 'Ir', 'Pt', 'Au', 'Hg'])[:,None]
        mask = (np.array(asemol.symbols)==metals).sum(axis=0)
        assert sum(mask)==1
        assert len(mask)==len(asemol)
        return mask
