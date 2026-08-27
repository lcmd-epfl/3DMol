#!/usr/bin/env python3

import pandas as pd

csv_path = 'data.csv'
df = pd.read_csv(csv_path, dtype={'idx': str})
print('*** {} total, {} ({:.0f}%) kept, {} ({:.0f}%) removed'.format(tot:=len(df), kept:=sum(df['is_point_group_noH_chiral']), kept/tot*100, removed:=tot-kept, removed/tot*100))
df = df[df['is_point_group_noH_chiral']]
df.drop(labels='is_point_group_noH_chiral', axis=1, inplace=True)
df.to_csv(csv_path, index=False)
