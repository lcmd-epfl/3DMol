#!/usr/bin/env python3

import os.path
import pandas as pd

results_path = 'results.csv'
raw_data_path = 'project_raw.csv'
raw_data_path = f'{os.path.dirname(__file__)}/../hypers/{raw_data_path}'

if not os.path.isfile(raw_data_path):
    raise RuntimeError(f'Result file {raw_data_path} is missing')

df = pd.read_csv(raw_data_path)
df = df[pd.isna(df['sweep_id'])]

results = []
for name in set(df.name):
    d = df[df['name'] == name].reset_index(drop=True)
    assert len(d) == 10, f'{name} {len(d)}'
    test_score = d.test_score
    results.append({
        'name': name,
        'target':    d.target_column[0],
        'geometry':  'xtb' if d.xtb[0] == True else 'default',
        'scope':     'local' if d.graph_mode[0]=='vector_masked' else 'global',
        'invariant': d.invariant[0],
        'mae_mean':  d.test_score.mean(),
        'mae_std':   d.test_score.std(),
        'rmse_mean': d.test_rmse.mean(),
        'rmse_std':  d.test_rmse.std()
        })

results = pd.DataFrame(results)
results.sort_values(by=['target', 'scope'], inplace=True, ignore_index=True)
results.to_csv(results_path, index=False)
