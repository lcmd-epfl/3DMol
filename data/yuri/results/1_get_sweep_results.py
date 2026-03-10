#!/usr/bin/env python3

import os.path
from itertools import chain
import pandas as pd

raw_data_path = 'sweep_best_runs.csv'
max_runs = 64
dataset = 'yuri'
project = "equireact/nequimol"
score_column = 'val_score_best'
drop_keys = ['num_epochs', 'CV iter', 'epoch', 'subset', 'random_baseline',
             'train loss', 'val_loss', 'val_score', 'classification']
meta_keys = ['run_id', 'sweep_id', 'name', 'val_score_best', 'splitter']
sweep_ids = ['lif8aeuy', 'yqyy9zgs', 'mly7kccj', 'j8flw45y', 'bozlc35j', '2swidcb1',
             '8so9j474', 'zy4w3ezs', 'xmn7nv76', 'r0ni4mgm', 'ggcl5g6t', 'nvz3xmao']


if not os.path.isfile(raw_data_path):

    import wandb

    summary_list = []
    api = wandb.Api()

    for sweep_id in sweep_ids:
        print(sweep_id)
        sweep = api.sweep(f'{project}/{sweep_id}')
        runs = [run for run in sweep.runs]
        assert len(runs) == max_runs

        runs = sorted(runs, key=lambda run: int(run.name.split('-')[-1]))
        best_run = sorted(runs, key = lambda run: float(run.summary[score_column]))[0]
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
    d['dataset']  = d['dataset'].replace('yuri', 'TMGSspinPlus')
    d['splitter'] = d['splitter'].replace('yuri', 'TMGSspinPlus')
    print(d)
    run_id = d['run_id']
    sweep_id = d['sweep_id']
    dataset = d['dataset']
    target = d['target_column']
    scope = 'local' if 'vector_masked'==d['graph_mode'] else 'global'
    name = f'{dataset.split(":")[-1]}-{target}-{scope}-{sweep_id}-{run_id}'
    print(name)
    with open(f"configs/config-{name}.dat", 'w') as f:
        print('#', ', '.join(f'{key}={d[key]}' for key in meta_keys), file=f)
        print('\n'.join(f'{key}\t{d[key]}' for key in sorted(d.keys()) if key not in meta_keys), file=f)
