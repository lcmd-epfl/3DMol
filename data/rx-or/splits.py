#!/usr/bin/env python3

from types import SimpleNamespace
import numpy as np

N = 132315
seed = 666
tr_frac = 0.8
nsplits = 10

np.random.seed(seed)
indices = np.random.permutation(np.arange(N))

te_frac = (1. - tr_frac) / 2
tr_size = round(tr_frac * len(indices))
te_size = round(te_frac * len(indices))

idx = SimpleNamespace(train=[None for i in range(nsplits)], test=[None for i in range(nsplits)], val=[None for i in range(nsplits)])

idx.test = np.array_split(indices, nsplits)
for i in range(nsplits):
    train_val = np.random.permutation(np.setdiff1d(indices, idx.test[i]))
    idx.train[i], idx.val[i] = np.split(train_val, [tr_size])

for i in range(nsplits):
    x = np.hstack((idx.train[i], idx.test[i], idx.val[i]))
    print(idx.train[i].shape, idx.test[i].shape, idx.val[i].shape, np.linalg.norm(np.sort(x)-np.sort(indices)))
print(np.linalg.norm(np.sort(np.hstack(idx.test))-np.sort(indices)))

for i in range(nsplits):
    np.savetxt(f'splits/test.{i}.dat', idx.test[i], fmt='%d')
    np.savetxt(f'splits/val.{i}.dat', idx.val[i], fmt='%d')
