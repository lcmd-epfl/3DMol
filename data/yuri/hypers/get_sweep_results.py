#!/usr/bin/env python3

import os.path
import pandas as pd
import wandb

raw_data_path = 'project_raw.csv'
dataset = 'yuri'

# download the data
if not os.path.isfile(raw_data_path):
    api = wandb.Api()
    runs = api.runs("equireact/nequimol")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        name_list.append({'run_id': run.id, 'sweep_id': (run.sweep.id if run.sweep is not None else None), 'name': run.name})
        summary_list.append({k: v for k,v in run.summary._json_dict.items() if not k.startswith('_')})
        config_list.append({k: v for k,v in run.config.items() if not k.startswith('_')})

    name_df = pd.DataFrame(name_list)
    summary_df = pd.DataFrame(summary_list)
    config_df = pd.DataFrame(config_list)
    assert len(set.intersection(set(name_df.columns), set(summary_df.columns), set(config_df.columns)))==0
    df = name_df.join(summary_df).join(config_df)
    df = df[df['dataset'] == dataset]
    df.to_csv(raw_data_path, index=False)
else:
    df = pd.read_csv(raw_data_path)

# split into sweeps
drop_keys = ['num_epochs', 'CV iter', 'epoch', 'train loss', 'val_loss', 'val_score', 'test_rmse', 'test_score', 'subset', 'xtb_subset', 'xtb', 'random_baseline']
meta_keys = ['run_id', 'sweep_id', 'name', 'val_score_best', 'splitter']
target_column = 'val_score_best'
max_num = 64

df.dropna(subset='sweep_id', inplace=True, ignore_index=True)
df.drop(labels=drop_keys, axis=1, inplace=True)

df_sweep = {sweep_id: df[df['sweep_id'] == sweep_id].iloc[::-1].reset_index(drop=True) for sweep_id in set(df.sweep_id)}

for sweep_id, d in df_sweep.items():

    target = d['target_column'][0]
    scope = 'local' if 'vector_masked' in list(d['graph_mode']) else 'global'
    name = f'{target}-{scope}-{sweep_id}'
    assert not d[target_column].isnull().any(), f'{name} contains empty error values - check if crashed'
    assert len(d) >= max_num, f'{name} contains not enough runs'

    d.to_csv(f"sweep-{name}.csv", index=False)

    d = d[:max_num]
    best = d.loc[d[target_column].idxmin()]

    with open(f"config-{name}.dat", 'w') as f:
        print('#', ', '.join(f'{key}={best[key]}' for key in meta_keys), file=f)
        print('\n'.join(f'{key}\t{best[key]}' for key in sorted(best.keys()) if key not in meta_keys), file=f)
