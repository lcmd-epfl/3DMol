import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem

def canonicalize(x):
    try:
        return Chem.CanonSmiles(x)
    except:
        return None


def check_elements(x):
    heavy_elements = ['I']
    rare_elements = ['P', 'Se']
    return len(set(a.GetSymbol() for a in Chem.MolFromSmiles(x).GetAtoms()).intersection(heavy_elements+rare_elements))==0


df = pd.read_table('data_0_raw.tsv')
df.dropna(subset='SMILES', inplace=True, ignore_index=True)
df = df[df['SMILES'].str.contains('@')].reset_index(drop=True)
df.drop_duplicates('Substance Identification: Reaxys Registry Number', keep='first', inplace=True, ignore_index=True)
df.drop_duplicates('SMILES', keep='first', inplace=True, ignore_index=True)
df['canon_SMILES'] = [*map(canonicalize, df.SMILES)]
df.dropna(subset='canon_SMILES', inplace=True, ignore_index=True)
df.drop_duplicates('canon_SMILES', keep='first', inplace=True, ignore_index=True)
df = df[df.apply(lambda x: check_elements(x['canon_SMILES']), axis=1)]

df_new = pd.DataFrame({'id':df['Substance Identification: Reaxys Registry Number'], 'SMILES':df['SMILES'], 'canon_SMILES':df['canon_SMILES'], 'specific_rotation': df['Optical Rotatory Power [deg]']})
df_new.to_csv('data_1_cleaned.csv', index=False)


