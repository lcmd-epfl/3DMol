import numpy as np

# split so each subset has only both enantiomers
# i.e. left/right are in the same sets
# the opposite to the second splitting

np.random.seed(666)
N = 512//2
n = N//10

x = np.random.choice(N, size=n*2, replace=False)
np.savetxt('idx_test_even.dat', np.concatenate((x[:n], x[:n]+N)), fmt='%d')
np.savetxt('idx_val_even.dat', np.concatenate((x[n:], x[n:]+N)), fmt='%d')



# split so each subset has only one enantiomer
# i.e. left/right are in different sets
# the opposite to the first splitting


np.random.seed(666)
N = 512   # both positive and negative samples
n = N//4  # number of samples for test/val

test_size = n
val_size = n
train_size = N - test_size - val_size

all_indices = np.random.permutation(np.arange(N//2))

train_positive_idx = all_indices[:train_size//2]
test_positive_idx = all_indices[train_size//2:train_size//2 + test_size//2]
val_positive_idx = all_indices[train_size//2 + test_size//2:]

train_negative_idx = np.concatenate((test_positive_idx + N//2, val_positive_idx + N//2))
test_negative_idx = train_positive_idx[:train_size//4] + N//2
val_negative_idx = train_positive_idx[train_size//4:] + N//2

train_indices = np.concatenate((train_positive_idx, train_negative_idx))
test_indices = np.concatenate((test_positive_idx, test_negative_idx))
val_indices = np.concatenate((val_positive_idx, val_negative_idx))

assert len(train_indices) == len(set(train_indices)) == train_size
assert len(test_indices) == len(set(test_indices)) == test_size
assert len(val_indices) == len(set(val_indices)) == val_size
assert len(set(train_indices).intersection(set(test_indices))) == 0
assert len(set(train_indices).intersection(set(val_indices))) == 0
assert len(set(test_indices).intersection(set(val_indices))) == 0

assert len(set(train_indices%(N//2))) == train_size
assert len(set(test_indices%(N//2))) == test_size
assert len(set(val_indices%(N//2))) == val_size

np.savetxt('idx_test_uneven.dat', test_indices, fmt='%d')
np.savetxt('idx_val_uneven.dat', val_indices, fmt='%d')
