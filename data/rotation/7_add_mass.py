import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.Descriptors

df = pd.read_csv('data.csv')
weights = [rdkit.Chem.Descriptors.MolWt(rdkit.Chem.MolFromSmiles(i)) for i in df.canon_SMILES]
weights_signed = np.sign(df.specific_rotation_computed.to_numpy()) * weights
df['weights'] = [rdkit.Chem.Descriptors.MolWt(rdkit.Chem.MolFromSmiles(i)) for i in df.canon_SMILES]
df['weights_signed'] = df.weights.to_numpy() * np.sign(df.specific_rotation_computed.to_numpy())
df.to_csv('data.csv', index=False)
