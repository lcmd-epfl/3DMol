from process.dataloader import MolDataset


class PropargReactants(MolDataset):
    def __init__(self, process=True, verbose=4,
                 xtb=False, noH=True, graph_method='smiles_mapped'):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/proparg/data_reactants.csv'
        self.processed_dir='data/proparg/processed/'
        self.smiles_column = 'smiles_mapped'
        self.id_column = 'xyz_id'
        self.target_column = 'Eafw'
        if xtb:
            files_dir='data/proparg/xyz-xtb/'
            geometry = 'xtb'
        else:
            files_dir='data/proparg/xyz/'
            geometry = 'dft'
        self.get_xyz_path = lambda idx: f'{files_dir}/{idx}.r.xyz'

        super().__init__(process=process, geometry=geometry, noH=noH,
                         graph_method=graph_method, verbose=verbose)
