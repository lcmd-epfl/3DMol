from process.dataloader import MolDataset


class dsC7O2H10nsd(MolDataset):
    def __init__(self, process=True, verbose=4, target_column='gap_Hartree',
                 noH=True, graph_method='smiles_loose'):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/qm9/dsC7O2H10nsd.csv'
        self.processed_dir='data/qm9/processed/'
        self.smiles_column = 'smiles'
        self.id_column = 'xyz'
        geometry = 'dft'
        self.get_xyz_path = lambda idx: f'data/qm9/{idx}'

        super().__init__(process=process, geometry=geometry, noH=noH,
                         target_column=target_column,
                         graph_method=graph_method, verbose=verbose)


class QM9(MolDataset):
    def __init__(self, process=True, verbose=4, target_column='gap_Hartree',
                 noH=True, graph_method='smiles_loose', check=True):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/qm9/dsgdb9nsd.csv'
        self.processed_dir='data/qm9/processed/'
        self.smiles_column = 'smiles'
        self.id_column = 'xyz'
        self.bad_indices = 'data/qm9/dsgdb9nsd_bad.txt'
        geometry = 'dft'
        self.get_xyz_path = lambda idx: f'data/qm9/{idx}'

        super().__init__(process=process, geometry=geometry, noH=noH,
                         target_column=target_column, check=check,
                         graph_method=graph_method, verbose=verbose)
