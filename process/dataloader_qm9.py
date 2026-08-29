from process.dataloader import MolDataset


class dsC7O2H10nsd(MolDataset):
    def __init__(self, *, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/qm9/dsC7O2H10nsd.csv'
        self.processed_dir='data/qm9/processed/'
        self.smiles_column = 'smiles'
        self.id_column = 'xyz'
        self.default_parameters = dict(geometry='dft', target_column='gap_Hartree', graph_method='smiles_loose')
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='dft':
            self.get_xyz_path = lambda idx: f'data/qm9/{idx}'

        super().__init__(bad_indices=None, verbose=verbose, check=check, **kwargs)


class QM9(MolDataset):
    def __init__(self, *, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/qm9/dsgdb9nsd.csv'
        self.processed_dir='data/qm9/processed/'
        self.smiles_column = 'smiles'
        self.id_column = 'xyz'
        bad_indices = 'data/qm9/dsgdb9nsd_bad.txt'
        self.default_parameters = dict(geometry='dft', target_column='gap_Hartree', graph_method='smiles_loose')
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='dft':
            self.get_xyz_path = lambda idx: f'data/qm9/{idx}'

        super().__init__(bad_indices=bad_indices, verbose=verbose, check=check, **kwargs)
