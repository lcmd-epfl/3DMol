#!/usr/bin/env python3

import os.path
from itertools import chain
import pandas as pd
import wandb

raw_data_path = 'sweep_best_runs.csv'
max_runs = 32
datasets = ['QM9Rotation']
project = "equireact/3dmol-rot"
score_column = 'val_score_best'
drop_keys = ['num_epochs', 'CV iter', 'epoch', 'subset',
             'train loss', 'val_loss', 'val_score', 'graph_method']
meta_keys = ['run_id', 'sweep_id', 'name', 'val_score_best', 'splitter']
sweep_ids_dict = {
'QM9Rotation': ['gwa4qt7v']
}

if not os.path.isfile(raw_data_path):

    summary_list = []
    api = wandb.Api()

    for dataset in datasets:
        if dataset not in sweep_ids_dict:
            runs = api.runs(project)
            sweep_ids = list(set([run.sweep.id for run in runs
                                               if run.config['dataset'].endswith(dataset) and run.sweep is not None]))
            print(dataset, sweep_ids)
        else:
            sweep_ids = sweep_ids_dict[dataset]

        for sweep_id in sweep_ids:
            sweep = api.sweep(f'{project}/{sweep_id}')
            runs = [run for run in sweep.runs]
            assert len(runs) >= max_runs

            runs = sorted(runs, key=lambda run: int(run.name.split('-')[-1]))[:max_runs]

            best_idx = -1 if runs[0].config['classification'] else 0
            best_run = sorted(runs, key = lambda run: float(run.summary[score_column]))[best_idx]
            d1 = {'run_id': best_run.id, 'sweep_id': sweep.id, 'name': best_run.name}
            d2 = {k: v for k, v in best_run.summary._json_dict.items() if not k.startswith('_')}
            d3 = {k: v for k, v in best_run.config.items() if not k.startswith('_')}
            assert len(set.intersection(set(d1.keys()), set(d2.keys()), set(d3.keys())))==0
            d1.update(d2)
            d1.update(d3)
            summary_list.append(d1)

    df = pd.DataFrame(summary_list)
    df.drop(labels=drop_keys, axis=1, inplace=True)
    df.to_csv(raw_data_path, index=False)
else:
    df = pd.read_csv(raw_data_path)

for _, d in df.iterrows():
    print(d)
    run_id = d['run_id']
    sweep_id = d['sweep_id']
    dataset = d['dataset']
    target = d['target_column']
    arch = d['arch']
    name = f'{dataset.split(":")[-1]}-{target}-{arch}-{sweep_id}-{run_id}'
    print(name)
    with open(f"configs/config-{name}.dat", 'w') as f:
        print('#', ', '.join(f'{key}={d[key]}' for key in meta_keys), file=f)
        print('\n'.join(f'{key}\t{d[key]}' for key in sorted(d.keys()) if key not in meta_keys), file=f)
