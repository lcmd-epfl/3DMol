import numpy as np
import pandas as pd
from tqdm import tqdm

# https://zenodo.org/records/13380412
data = np.load('qm9-or.npy', allow_pickle=True)

elements = np.array(['H', 'C', 'N', 'O', 'F'])

data_dict = {
        'idx' : [],
        'rot633' : [],
        'rot589' : [],
        'rot355' : [],
        }

for i in tqdm(data):
    idx =i['index']
    rot633, rot589, rot355 = i['rotation']
    coords = i['xyz'][:,:3]
    atoms = i['xyz'][:,3:]
    atoms = elements[np.where(atoms==1.0)[1]]

    data_dict['idx'   ].append(idx)
    data_dict['rot633'].append(rot633)
    data_dict['rot589'].append(rot589)
    data_dict['rot355'].append(rot355)

    with open(f'xyz/{idx}.xyz', 'w') as f:
        print(len(atoms), file=f)
        print(idx, file=f)
        for q, r in zip(atoms, coords):
            print(q, *r, file=f)

df = pd.DataFrame(data_dict)
df.to_csv('data_raw.csv', index=False)
