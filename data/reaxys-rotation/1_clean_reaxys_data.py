#!/usr/bin/env python3

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


def check_central_chirality(x):
    centers = [i[1] for i in Chem.FindMolChiralCenters(Chem.MolFromSmiles(x), includeUnassigned=True)]
    return ('?' not in centers) and (len(centers)>0) # removes also axis/plane/helix but ok for now


def merge_smiles(df):
    # check that the same ID has the same canonical smiles
    for name, group in df.groupby(by=['id']):
        if len(group)>1:
            assert len(set(group['canon_SMILES']))==1
    # set the lowest ID to the rows with the same canonical smiles
    for name, group in df.groupby(by=['canon_SMILES']):
        if len(group)>1:
            df.loc[group.index, 'id'] = min(group['id'])
    df.sort_values('id', inplace=True, ignore_index=True)


def get_mirror(x):
    return canonicalize(x.replace("@@", "__DOUBLE_AT__").replace("@", "@@").replace("__DOUBLE_AT__", "@"))


def clean_duplicates(df):
    # remove duplicates and enantiomers
    for name, group in df.groupby(by=['both_SMILES']):
        if len(group)==1:
            continue
        ref_entry = group.iloc[0]
        ref_smi = ref_entry['canon_SMILES']
        rotation = [ref_entry['specific_rotation']]

        for i in range(1, len(group)):
            entry = group.iloc[i]
            smi, mirror_smi, rot = entry['canon_SMILES'], entry['canon_SMILES_mirror'], entry['specific_rotation']
            if smi==ref_smi:
                rotation.append(rot)
            elif mirror_smi==ref_smi:
                rotation.append(-rot)
            else:
                raise RuntimeError

        rotation = np.array(rotation)
        if sum(rotation>0)==sum(rotation<0):
            mean_rotation = None
        else:
            mean_rotation = np.median(rotation)
        df.loc[group.index[0], 'specific_rotation'] = mean_rotation  # will drop all entries but first

    df.drop_duplicates('both_SMILES', keep='first', inplace=True, ignore_index=True)
    df.dropna(subset='specific_rotation', inplace=True, ignore_index=True)


df = pd.read_table('data_0_raw.tsv')
df.dropna(subset='SMILES', inplace=True, ignore_index=True)
df = df[df['SMILES'].str.contains('@')].reset_index(drop=True)
df.rename(columns={'Substance Identification: Reaxys Registry Number': 'id',
                   'Optical Rotatory Power [deg]': 'specific_rotation'}, inplace=True)
df['canon_SMILES'] = [*map(canonicalize, df.SMILES)]
df.dropna(subset='canon_SMILES', inplace=True, ignore_index=True)
df = df[df.apply(lambda x: check_central_chirality(x['canon_SMILES']), axis=1)]
df = df[df.apply(lambda x: check_elements(x['canon_SMILES']), axis=1)]
df['canon_SMILES_mirror'] = [*map(get_mirror, df.canon_SMILES)]
df['both_SMILES'] = [*map(lambda x: '.'.join(sorted(x)), zip(df['canon_SMILES'], df['canon_SMILES_mirror']))]

df['nmol'] = df['SMILES'].map(lambda x: x.count('.')+1)
bad_mask = df['nmol']>1
#print('more than 1 mol: ', sum(bad_mask))
df = df[~bad_mask].reset_index(drop=True)
df.drop(['nmol'], axis=1, inplace=True)

merge_smiles(df)
clean_duplicates(df)

df_new = pd.DataFrame({key: df[key] for key in ['id', 'SMILES', 'canon_SMILES', 'specific_rotation']})
df_new.to_csv('data_1_cleaned.csv', index=False)
