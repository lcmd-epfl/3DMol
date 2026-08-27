#!/usr/bin/env python3

import pandas as pd
from collections import Counter
from rdkit import Chem
from tqdm import trange

csv_path = 'data.csv'

df = pd.read_csv(csv_path, dtype={'idx': str})

for i in trange(len(df), disable=False):
    mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
    element_counts = dict(Counter([atom.GetSymbol() for atom in mol.GetAtoms()]))
    for q, n in element_counts.items():
        df.loc[i, f'element:{q}'] = n
df.fillna(0, inplace=True)
df.drop('element:H', axis=1, inplace=True)

elements = list(filter(lambda x: x.startswith('element:'),  df.columns.to_list()))

def find_bad():
    bad_elements = []
    nq = {q: sum(df[q]) for q in elements}
    total = sum(n for n in nq.values())
    for q in elements:
        nmols = sum(df[q]>0)
        fmols = nmols/len(df)*100.0
        if fmols < 0.5:
            bad_elements.append(q)
            bad = '*'
        else:
            bad = ''
        print('{}\t{:.0f}\t({:.1e}%)\t{}\t({:.1e}%)\t{}'.format(q, nq[q], nq[q]/total*100, nmols, fmols, bad))
    return bad_elements

bad_elements = find_bad()

for q in bad_elements:
    df = df[df[q]==0].reset_index(drop=True)

find_bad()

df.drop(elements, axis=1, inplace=True)
df.to_csv(csv_path, index=False)
