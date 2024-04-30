from process.dataloader import MolDataset


class dsC7O2H10nsd(MolDataset):
    def __init__(self, process=True, verbose=4,
                 xtb=False, noH=True, graph_method='smiles_loose'):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/qm9/dsC7O2H10nsd.csv'
        self.processed_dir='data/qm9/processed/'
        self.smiles_column = 'smiles'
        self.id_column = 'xyz'
        self.target_column = 'gap_Hartree'
        geometry = 'dft'
        self.get_xyz_path = lambda idx: f'data/qm9/{idx}'

        super().__init__(process=process, geometry=geometry, noH=noH,
                         graph_method=graph_method, verbose=verbose)
