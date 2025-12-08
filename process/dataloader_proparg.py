from process.dataloader import MolDataset


class PropargReactants(MolDataset):
    def __init__(self, extra_args=None, target_column=None,
                 geometry=None, graph_method=None,
                 verbose=4, check=False, **kwargs):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/proparg/data_reactants.csv'
        self.processed_dir='data/proparg/processed/'
        self.smiles_column = 'smiles_mapped'
        self.id_column = 'xyz_id'
        self.default_parameters = dict(geometry='dft', target_column='Eafw', graph_method='smiles_mapped')
        self.parameters = self.get_parameters(locals())

        if self.parameters.geometry=='xtb':
            files_dir='data/proparg/xyz-xtb/'
        elif self.parameters.geometry=='dft':
            files_dir='data/proparg/xyz/'
        else:
            raise RuntimeError(f'unknown geometry {geometry}')
        self.get_xyz_path = lambda idx: f'{files_dir}/{idx}.r.xyz'

        super().__init__(bad_indices=bad_indices, verbose=verbose, check=check, **kwargs)
