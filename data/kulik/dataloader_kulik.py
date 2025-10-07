import numpy as np
from process.dataloader import MolDataset


class tmPHOTO(MolDataset):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 classification=False,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):

        self.version = 2  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.processed_dir='data/kulik/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='xtb', target_column='gap', graph_method='torchchem_v1',
                                       _dl_extra_csv_path='data/kulik/tmPHOTO_property.csv')
        bad_indices = None
        self.parameters = self.get_parameters(locals() | (extra_args if extra_args else {}))

        self.csv_path = self.parameters._dl_extra_csv_path

        if self.parameters.geometry=='xtb':
            self.get_xyz_path = lambda idx: f'data/kulik/tmPHOTO-xyz/{idx}.xyz'

        super().__init__(process=process, noH=noH,
                         classification=classification,
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
                 extra_args=None,
                 classification=False,
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
                         classification=classification,
                         check=check, bad_indices=bad_indices,
                         verbose=verbose)


    def get_local_mask(self, asemol, **kwargs):
        metals = np.array(['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Mo', 'Ru'])[:,None]
        mask = (np.array(asemol.symbols)==metals).sum(axis=0)
        assert sum(mask)==1
        assert len(mask)==len(asemol)
        return mask


class _Octa(MolDataset):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 classification=False,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False, csv_path=None):

        self.version = 2  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path = f'data/kulik/{csv_path}'
        self.processed_dir='data/kulik/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        self.default_parameters = dict(geometry='default', target_column=None, graph_method='torchchem_v1')
        bad_indices = None
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='default':
            self.get_xyz_path = lambda idx: f'data/kulik/OctaKulik-xyz/{idx}.xyz'

        super().__init__(process=process, noH=noH,
                         classification=classification,
                         check=check, bad_indices=bad_indices,
                         verbose=verbose)


    def get_local_mask(self, asemol, **kwargs):
        metals = np.array(['Cr', 'Mn', 'Fe', 'Co'])[:,None]
        mask = (np.array(asemol.symbols)==metals).sum(axis=0)
        assert sum(mask)==1
        assert len(mask)==len(asemol)
        return mask


class OctaFull(_Octa):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 classification=False,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):
        super().__init__(process=process, verbose=verbose, extra_args=extra_args,
                         classification=classification,
                         target_column=target_column, geometry=geometry, graph_method=graph_method,
                         noH=noH, check=check, csv_path='OctaKulik_property_HOMO_LUMO_gap.csv')


class OctaLow(_Octa):
    def __init__(self, process=True, verbose=4,
                 extra_args=None,
                 classification=False,
                 target_column=None, geometry=None, graph_method=None,
                 noH=True, check=False):
        super().__init__(process=process, verbose=verbose, extra_args=extra_args,
                         classification=classification,
                         target_column=target_column, geometry=geometry, graph_method=graph_method,
                         noH=noH, check=check, csv_path='OctaKulik_property_splitting.csv')
