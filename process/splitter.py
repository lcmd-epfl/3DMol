import numpy as np

indices_names = ('train', 'test', 'val')


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
    if subset:
        raise RuntimeError('subset option incompatible with test/train/val indices file')

    fnames = dict([entry.split(':') for entry in splitter.split(';')])
    if len(fnames)!=len(set(fnames.keys())):
        raise RuntimeError(f'repeated indices name(s) {tuple(fnames.keys())}')
    if not set(fnames.keys()).issubset(set(indices_names)):
        raise RuntimeError(f'bad indices name(s) {tuple(fnames.keys())} - should be in {indices_names}')

    print(fnames)

    indices_dict = {key: np.load(fnames[key]) if fnames[key].endswith('.npy') else np.loadtxt(fnames[key], dtype=int, ndmin=1) for key in fnames}

    # only test
    if len(fnames)==1 and next(iter(fnames.keys()))=='test':
        te_indices = indices_dict['test']
        if len(te_indices) != te_size:
            raise RuntimeError(f'Fix the training set size so the requested test set size ({te_size}) corresponds to the test indices file size ({len(te_indices)})')
        if len(np.intersect1d(te_indices, indices))<len(te_indices):
            raise RuntimeError('bad test indices')
        indices_notest = np.array([i for i in indices if i not in te_indices])
        tr_indices, val_indices = np.split(indices_notest, [tr_size])

    # only test and val
    elif len(fnames)==2 and set(fnames.keys())=={'test', 'val'}:
        te_indices = indices_dict['test']
        val_indices = indices_dict['val']
        if len(np.intersect1d(te_indices, val_indices)):
            raise RuntimeError('validation and test sets overlap')
        if len(np.intersect1d(te_indices, indices))<len(te_indices):
            raise RuntimeError('bad test indices')
        if len(np.intersect1d(val_indices, indices))<len(val_indices):
            raise RuntimeError('bad val indices')
        if len(te_indices) != te_size:
            print(f'Fix the training set size so the requested test set size ({te_size}) corresponds to the test indices file size ({len(te_indices)})')
        tr_indices = np.array([i for i in indices if i not in np.concatenate((te_indices, val_indices))])
        if len(tr_indices) != tr_size:
            print('bad training set size')  # TODO

    else:
        raise NotImplementedError

    return tr_indices, te_indices, val_indices


def split_dataset(data, splitter, tr_frac, subset=None):
    '''
    1) seed `np.random` and `random` before calling this fn
    2) use the output indices with np.arrays, lists, df.iloc[]
    '''
    indices = np.arange(data.nmols)
    np.random.shuffle(indices)
    if subset:
        indices = indices[:subset]
        assert len(indices) == subset, "lost data in subset"

    te_frac = (1. - tr_frac) / 2
    tr_size = round(tr_frac * len(indices))
    te_size = round(te_frac * len(indices))

    if splitter == 'random':
        print("Using random splits")
        tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])

    elif splitter in {'yasc', 'ydesc'}:  # splits based on the target value
        print(f"Using target-based splits ({'ascending' if splitter=='yasc' else 'descending'} order)")
        tr_indices, te_indices, val_indices = get_y_splits(data, splitter, indices, tr_size, te_size)

    elif splitter in {'sizeasc', 'sizedesc'}:
        print(f"Splitting based on molecular size ({'ascending' if splitter=='sizeasc' else 'descending'} order)")
        tr_indices, te_indices, val_indices = get_size_splits(data, splitter, indices, tr_size, te_size)

    elif sum(splitter.startswith(f'{i}:') for i in indices_names):
        print("Using indices from file")
        tr_indices, te_indices, val_indices = get_test_file_splits(splitter, indices, tr_size, te_size, subset)

    else:
        raise RuntimeError

    return tr_indices, te_indices, val_indices, indices
