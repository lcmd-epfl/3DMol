import re
import numpy as np
import pandas as pd
from chemprop.data.utils import get_data_from_smiles
from process.scaffold import scaffold_split


def remove_atom_map_number_manual(smi):
    return re.sub(':[0-9]+', '', smi)


def get_scaffold_splits(data, indices=None, sizes=(0.8, 0.1, 0.1)):
    smiles = pd.read_csv(data.csv_path, index_col=0)[data.smiles_column].to_numpy()
    smiles = smiles[indices]
    chemprop_dataset = get_data_from_smiles([[remove_atom_map_number_manual(x)] for x in smiles])
    train_idx, test_idx, val_idx = scaffold_split(chemprop_dataset, sizes=sizes, balanced=True)
    return indices[train_idx], indices[test_idx], indices[val_idx]


def get_y_splits(data, splitter, indices, tr_size, te_size):
    y = data.labels.numpy()
    idx4idx = np.argsort(y[indices])
    if splitter == 'ydesc':
        idx4idx = idx4idx[::-1]
    indices = indices[idx4idx]
    tr_indices, val_indices, te_indices = np.split(indices, [tr_size, tr_size+te_size])
    np.random.shuffle(tr_indices)
    np.random.shuffle(te_indices)
    np.random.shuffle(val_indices)
    return tr_indices, te_indices, val_indices


def get_size_splits(data, splitter, indices, tr_size, te_size):
    """
    train-test split based on molecule size:
    train on smaller molecules and test on larger molecules
    or vice versa

    Args:
        data: dataset object
        splitter: sizeasc / sizedesc
        indices: for subset of data
        tr_size: float
        te_size: float

    Returns:
        tr_indices, te_indices, val_indices: tuple of list/arr of indices
        """

    mol_sizes = np.array([g.num_nodes for g in data.mol_graphs])

    idx4idx = np.argsort(mol_sizes[indices])
    if splitter == 'sizedesc':
        idx4idx = idx4idx[::-1]
    indices = indices[idx4idx]

    tr_indices, val_indices, te_indices = np.split(indices, [tr_size, tr_size+te_size])
    np.random.shuffle(tr_indices)
    np.random.shuffle(te_indices)
    np.random.shuffle(val_indices)
    return tr_indices, te_indices, val_indices


def get_test_file_splits(splitter, indices, tr_size, te_size, subset):
    fname = splitter[5:]
    if subset:
        raise RuntimeError('subset option incompatible with test indices file')
    try:
        te_indices = np.load(fname) if fname.endswith('.npy') else np.loadtxt(fname)
    except:
        raise RuntimeError
    if len(te_indices) != te_size:
        raise RuntimeError(f'Fix the training set size so the requested test set size ({te_size}) corresponds to the test indices file size ({len(te_indices)})')
    indices_notest = np.array([i for i in indices if i not in te_indices])
    tr_indices, val_indices = np.split(indices_notest, [tr_size])
    return tr_indices, te_indices, val_indices


def split_dataset(data, splitter, tr_frac, subset=None):
    '''
    1) seed `np.random` and `random` before calling this fn
    2) use the output indices with np.arrays, lists, df.iloc[]
    '''
    indices = np.arange(data.nmols)
    len_before = len(indices)
    np.random.shuffle(indices)
    len_after = len(indices)
    assert len_before == len_after, "lost data in shuffle"
    if subset:
        indices = indices[:subset]
        assert len(indices) == subset, "lost data in subset"

    te_frac = (1. - tr_frac) / 2
    tr_size = round(tr_frac * len(indices))
    te_size = round(te_frac * len(indices))
    va_size = len(indices) - tr_size - te_size

    if splitter == 'random':
        print("Using random splits")
        tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])

    elif splitter in ['yasc', 'ydesc']: # splits based on the target value
        print(f"Using target-based splits ({'ascending' if splitter=='yasc' else 'descending'} order)")
        tr_indices, te_indices, val_indices = get_y_splits(data, splitter, indices, tr_size, te_size)

    elif splitter in ['sizeasc', 'sizedesc']:
        print(f"Splitting based on molecular size ({'ascending' if splitter=='sizeasc' else 'descending'} order)")
        tr_indices, te_indices, val_indices = get_size_splits(data, splitter, indices, tr_size, te_size)

    elif splitter == 'scaffold':
        print("Using scaffold splits")
        tr_indices, te_indices, val_indices = get_scaffold_splits(data, indices, sizes=(tr_frac, 1-(tr_frac+te_frac), te_frac))

    elif splitter.startswith('test:'):
        print("Using test indices from file")
        tr_indices, te_indices, val_indices = get_test_file_splits(splitter, indices, tr_size, te_size, subset)


    else:
        raise RuntimeError

    return tr_indices, te_indices, val_indices, indices
