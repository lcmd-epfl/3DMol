# https://gist.github.com/tdudgeon/b061dc67f9d879905b50118408c30aac

import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0)
    return list(ids)


def calc_energy(mol, conformerId, minimizeIts):
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId)
    if ff is None:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conformerId)
    ff.Initialize()
    ff.CalcEnergy()
    ff.Minimize(maxIts=minimizeIts)
    return ff.CalcEnergy()


df = pd.read_csv('data_1_cleaned.csv')

for i in tqdm(range(len(df))):

    smi = df.SMILES[i]
    mol = Chem.MolFromSmiles(smi)
    chiral_smiles = [i[1] for i in Chem.FindMolChiralCenters(mol)]

    m = Chem.AddHs(mol)
    conformerIds = gen_conformers(m, numConfs=16)
    if len(conformerIds)==0:
        print(f'{i} {df.id[i]}')
        continue
    conformerPropsDict = {conformerId: calc_energy(m, conformerId, 128) for conformerId in conformerIds}
    confId = sorted(conformerPropsDict, key=conformerPropsDict.get)[0]

    AllChem.AssignStereochemistryFrom3D(m, confId=confId)
    chiral_3d = [i[1] for i in Chem.FindMolChiralCenters(m)]
    if sorted(chiral_smiles)!=sorted(chiral_3d):
        print(f'{i} {df.id[i]} {chiral_smiles}, {chiral_3d}')
        continue

    Chem.MolToXYZFile(m, f'xyz/{df.id[i]}.xyz', confId)
