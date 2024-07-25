from process.dataloader import MolDataset


class Yuri(MolDataset):
    def __init__(self, process=True, verbose=4, target_column='gap_Hartree',
                 noH=True, graph_method='torchchem_v1', check=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.csv_path='data/yuri/refcode_properties_2300.csv'
        self.processed_dir='data/yuri/processed/'
        self.smiles_column = None
        self.id_column = 'refcode'
        bad_indices = None
        geometry = 'dft'
        self.get_xyz_path = lambda idx: f'data/yuri/0-XYZS/{idx}.xyz'

        super().__init__(process=process, geometry=geometry, noH=noH,
                         target_column=target_column, check=check,
                         bad_indices=bad_indices,
                         graph_method=graph_method, verbose=verbose)
