
import numpy as np
from sklearn.model_selection import KFold


# random split for 3612 geometries
X_shape = 1806*2  #HS or LS optimized geometries
indices = np.arange(X_shape)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(kf.split(indices)):

    dir_path = "0-random_3612"
    print(dir_path, i, len(train_idx), len(test_idx))
    with open(f"{dir_path}/{i}_train_indices.txt", "w") as f:
        for tr_idx in train_idx:
            f.write(f"{tr_idx}\n")

    with open(f"{dir_path}/{i}_test_indices.txt", "w") as f:
        for te_idx in test_idx:
            f.write(f"{te_idx}\n")

###########################################################
X_shape = 1806  #only LS optimized geometries
indices = np.arange(X_shape)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(kf.split(indices)):

    dir_path = "1-random_1806"
    print(dir_path, i, len(train_idx), len(test_idx))
    with open(f"{dir_path}/{i}_train_indices.txt", "w") as f:
        for tr_idx in train_idx:
            f.write(f"{tr_idx}\n")

    with open(f"{dir_path}/{i}_test_indices.txt", "w") as f:
        for te_idx in test_idx:
            f.write(f"{te_idx}\n")

    dir_path = "2-HS_LS_same_fold_3612"
    with open(f"{dir_path}/{i}_train_indices.txt", "w") as f:
        for tr_idx in train_idx:
            f.write(f"{(2*tr_idx)}\n")
            f.write(f"{(2*tr_idx+1)}\n")

    with open(f"{dir_path}/{i}_test_indices.txt", "w") as f:
        for te_idx in test_idx:
            f.write(f"{(2*te_idx)}\n")
            f.write(f"{(2*te_idx+1)}\n")
