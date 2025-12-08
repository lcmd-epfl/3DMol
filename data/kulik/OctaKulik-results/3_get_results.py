#!/usr/bin/env python3

import time
import os.path
from itertools import chain
import pandas as pd
import wandb

results_path = 'results.csv'
raw_data_path = 'cv_runs.csv'
datasets = ['OctaFull', 'OctaLow']
project = "equireact/3dmol-TMC-benchmark"
drop_keys = ['CV iter', 'subset', 'random_baseline',
             'train loss', 'val_loss', 'val_score']

lost_ids = []

if not os.path.isfile(raw_data_path):
    api = wandb.Api(timeout=20)
    runs = api.runs(project)
    runs_lost = (api.run(f"{project}/{idx}") for idx in lost_ids)

    summary_list = []
    for run in chain(runs, runs_lost):
        if run.config['dataset'].split(':')[-1] not in datasets or run.sweep is not None:
            continue

        d1 = {'run_id': run.id, 'name': run.name}
        d2 = {k: v for k, v in run.summary._json_dict.items() if not k.startswith('_')}
        d3 = {k: v for k, v in run.config.items() if not k.startswith('_')}
        assert len(set.intersection(set(d1.keys()), set(d2.keys()), set(d3.keys())))==0
        print(d1)
        d1.update(d2)
        d1.update(d3)
        summary_list.append(d1)
        time.sleep(0.1)

    df = pd.DataFrame(summary_list)
    df.drop(labels=drop_keys, axis=1, inplace=True)
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.to_csv(raw_data_path, index=False)
else:
    df = pd.read_csv(raw_data_path)


results = []
for name in sorted(set(df.name)):
    d = df[df['name'] == name].reset_index(drop=True)
    if name.startswith('cv10'):
        assert len(d) == 10, f'{name} {len(d)}'
        assert len(set(d['splitter'])) == 10, f'{name}'
    results.append({
        'name': name,
        'dataset':   d.dataset[0].split(':')[-1],
        'target':    d.target_column[0],
        'geometry':  d.geometry[0],
        'scope':     'local' if d.graph_mode[0]=='vector_masked' else 'global',
        'invariant': d.invariant[0],
        'mae_mean':  d.test_score.mean(),
        'mae_std':   d.test_score.std(),
        'rmse_mean': d.test_rmse.mean(),
        'rmse_std':  d.test_rmse.std()
        })

results = pd.DataFrame(results)
results.sort_values(by=['dataset', 'target', 'scope'], inplace=True, ignore_index=True)
results.to_csv(results_path, index=False)
