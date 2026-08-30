import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.Descriptors

df = pd.read_csv('data_converged.csv')
mass = np.array([rdkit.Chem.Descriptors.MolWt(rdkit.Chem.MolFromSmiles(i)) for i in df.canon_SMILES])

def power(alpha, mass):
    return alpha * mass / 100.0

exp = df['specific_rotation'].to_numpy()
exp_power = power(exp, mass)
df['specific_rotation_sign'] = np.sign(exp)
df['specific_rotation_abs'] = np.abs(exp)
df['specific_rotation_power'] = exp_power
df['specific_rotation_power_abs'] = np.abs(exp_power)

comp = df['specific_rotation_computed'].to_numpy()
comp_power = power(comp, mass)
df['specific_rotation_computed_sign'] = np.sign(comp)
df['specific_rotation_computed_abs'] = np.abs(comp)
df['specific_rotation_computed_power'] = comp_power
df['specific_rotation_computed_power_abs'] = np.abs(comp_power)

df.to_csv('data_converged.csv', index=False)
