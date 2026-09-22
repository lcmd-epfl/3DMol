from types import SimpleNamespace
import itertools
import warnings
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
    return SimpleNamespace(train=tr_indices, test=te_indices, val=val_indices)


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
    return SimpleNamespace(train=tr_indices, test=te_indices, val=val_indices)


def get_file_splits(splitter, indices, tr_size, te_size):

    fnames = dict([entry.split(':') for entry in splitter.split(';')])
    if len(fnames)!=len(set(fnames.keys())):
        raise RuntimeError(f'repeated indices name(s) {tuple(fnames.keys())}')
    if not set(fnames.keys()).issubset(set(indices_names)):
        raise RuntimeError(f'bad indices name(s) {tuple(fnames.keys())} - should be in {indices_names}')

    print(fnames)

    split = {key: np.load(fnames[key]) if fnames[key].endswith('.npy') else np.loadtxt(fnames[key], dtype=int, ndmin=1) for key in fnames}

    if len(fnames)==1 and next(iter(fnames.keys()))=='test':
        indices_notest = np.array([i for i in indices if i not in split['test']])
        split['train'], split['val'] = np.split(indices_notest, [tr_size])
    elif len(fnames)==2 and set(fnames.keys())=={'test', 'val'}:
        split['train'] = np.array([i for i in indices if i not in split['test'] and i not in split['val']])
    elif len(fnames)==3 and set(fnames.keys())==set(indices_names):
        pass
    else:
        raise NotImplementedError

    for (name1, idx1), (name2, idx2) in itertools.combinations(split.items(), 2):
        if len(dup:=np.intersect1d(idx1, idx2)):
            raise RuntimeError(f'{name1} and {name2} sets overlap: {dup}')

    for name, idx in split.items():
        if len(diff:=np.setdiff1d(idx, indices)):
            msg = f'bad {name} indices: {diff}'
            raise RuntimeError(msg)

    for name, size in [('test', te_size), ('train', tr_size)]:
        if len(split[name]) != size:
            msg = f'The requested {name} set size ({size}) does not correspond to the {name} indices file size ({len(split[name])})'
            warnings.warn(msg, stacklevel=2)

    return SimpleNamespace(**split)


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
        split = SimpleNamespace(train=tr_indices, test=te_indices, val=val_indices)

    elif splitter in {'yasc', 'ydesc'}:  # splits based on the target value
        print(f"Using target-based splits ({'ascending' if splitter=='yasc' else 'descending'} order)")
        split = get_y_splits(data, splitter, indices, tr_size, te_size)

    elif splitter in {'sizeasc', 'sizedesc'}:
        print(f"Splitting based on molecular size ({'ascending' if splitter=='sizeasc' else 'descending'} order)")
        split = get_size_splits(data, splitter, indices, tr_size, te_size)

    elif sum(splitter.startswith(f'{i}:') for i in indices_names):
        print("Using indices from file")
        if subset:
            raise RuntimeError('subset option incompatible with test/train/val indices file')
        split = get_file_splits(splitter, indices, tr_size, te_size)

    else:
        msg = f'Unknow splitter: {splitter}'
        raise NotImplementedError(msg)

    split.n = len(indices)
    return split
