#!/usr/bin/env python3

import os
import pandas as pd
from tqdm import trange
from collections import Counter
from rdkit import Chem
from tqdm import trange

csv_path = 'data_converged.csv'
csv_clean_path = 'data.csv'

# remove achiral
df = pd.read_csv(csv_path, dtype={'idx': str})
bad_mask = ~(df['is_point_group_chiral'] & df['is_point_group_noH_chiral'])
print('total mols:   ', len(df))
print('achiral:      ', sum(~df['is_point_group_chiral']))
print('achiral w/o H:', sum(~df['is_point_group_noH_chiral']))
print('total bad:    ', sum(bad_mask))
df = df[~bad_mask].reset_index(drop=True)
df.drop(['is_point_group_chiral', 'is_point_group_noH_chiral'], axis=1, inplace=True)
print()

# remove rare elements
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
        if fmols < 5.0: ## different from the big set
            bad_elements.append(q)
            bad = '*'
        else:
            bad = ''
        print('{}\t{:.0f}\t({:.1e}%)\t{}\t({:.1e}%)\t{}'.format(q, nq[q], nq[q]/total*100, nmols, fmols, bad))
    return bad_elements

bad_elements = find_bad()
print()

for q in bad_elements:
    df = df[df[q]==0].reset_index(drop=True)

find_bad()
df.drop(elements, axis=1, inplace=True)

df.to_csv(csv_clean_path, index=False)
