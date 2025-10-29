import os
import sys
from datetime import datetime
import argparse
import pprint
import wandb
from train import train, Logger


def train_wrapper():
    with wandb.init(config=None):
        args = wandb.config
        try:
            train(run_dir, logname, project, wandb_name, args,
                  device='cuda', num_epochs=args.num_epochs, checkpoint=None,
                  verbose=False, print_predictions=False, eval_on_test=False,
                  sweep=True, print_repr=False,
                  dataset=args.dataset, process=False, noH=args.noH,
                  geometry=args.geometry,
                  target_column=args.target_column, features=args.features,
                  splitter=args.splitter, subset=args.subset, training_fractions=[args.train_frac], CV=1)
        except Exception as e:
            print(e)
            pass


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='yuri', help='dataset')
parser.add_argument('-t', '--target', default='gap', help='target column')
parser.add_argument('--id', default=None, help='sweep id if continue a sweep')
parser.add_argument('-n', '--num', default=64, help='number of runs')
parser.add_argument('--equivariant', action='store_true', help='use equivariant model (default: invariant)')
parser.add_argument('--local', action='store_true', help='use local (masked) model')
script_args = parser.parse_args()

features = {
            'yuri': 'torchchem_v1',
            'tmSCO': 'torchchem_v1',
            'tmPHOTO': 'torchchem_v1',
            'OctaFull': 'torchchem_v1',
            'OctaLow': 'torchchem_v1',
            }
geometry = {
        'yuri': 'default',
        'tmSCO': 'xtb',
        'tmPHOTO': 'xtb',
        'OctaFull': 'default',
        'OctaLow': 'default',
        }
epochs = {
        'yuri': 128,
        'tmSCO': 128,
        'tmPHOTO': 128,
        'OctaFull': 128,
        'OctaLow': 128,
        }
target_columns_good = {
        'yuri': ('HOMO', 'LUMO', 'gap', 'dipole_moment_Debye', 'splitting', 'hirshfeld'),
        'tmSCO': ('HOMO', 'LUMO', 'gap', 'dipole_moment_Debye', 'Dispersion_E', 'Metal_q', 'Polarizability'),
        'tmPHOTO': ('HOMO', 'LUMO', 'gap', 'dipole_moment_Debye', 'Dispersion_E', 'Metal_q', 'Polarizability'),
        'OctaFull': ('HOMO', 'LUMO', 'gap'),
        'OctaLow': ('splitting'),
        }
splitter = {
        'yuri': 'test:data/yuri/splits/0_test_indices.txt',
        'tmSCO': 'test:data/kulik/tmSCO-splits/tmSCO_0_test_indices.npy',
        'tmPHOTO': 'test:data/kulik/tmPHOTO-splits/tmPHOTO_0_test_indices.npy',
        'OctaFull': 'test:data/kulik/OctaKulik-splits/train_valid_3612/0_test_indices.txt',
        'OctaLow': 'test:data/kulik/OctaKulik-splits/train_valid_1806/0_test_indices.txt',
        }
dataset_full = {
        'yuri': 'data/yuri/dataloader_yuri.py:TMGSspinPlus',
        'tmSCO': 'data/kulik/dataloader_kulik.py:tmSCO',
        'tmPHOTO': 'data/kulik/dataloader_kulik.py:tmPHOTO',
        'OctaFull': 'data/kulik/dataloader_kulik.py:OctaFull',
        'OctaLow': 'data/kulik/dataloader_kulik.py:OctaLow',
        }
train_frac = {
    'yuri': 0.8,
    'tmSCO': 0.798,
    'tmPHOTO': 0.8,
    'OctaFull': 0.599,
    'OctaLow': 0.599,
    }
project = 'nequimol'

dataset = script_args.dataset
local = script_args.local
target_column = script_args.target
if target_column not in target_columns_good[dataset]:
    raise RuntimeError
run_dir = f'sweep_{dataset}_{target_column}_{datetime.now().strftime("%y%m%d-%H%M%S.%f")}'
if not os.path.exists(run_dir):
    try:
        os.makedirs(run_dir)
    except:
        pass
logname = 'sweep'

logpath = os.path.join(run_dir, f'{logname}.log')
sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

wandb.login()

metric = { 'name': 'val_score_best', 'goal': 'minimize' }
sweep_config = { 'method': 'bayes', 'metric': metric, 'name': f'{dataset}_{target_column}_{local=}' }

parameters_dict = {
    'distance_emb_dim': { 'values': [16, 32, 48, 64] },
    'dropout_p': { 'values': [0.0, 0.05, 0.1] },
    'max_neighbors': { 'values': [10, 25, 50] },
    'n_conv_layers': { 'values': [2, 3] },
    'n_s': { 'values': [16, 32, 48, 64] },
    'radius': { 'values' : [2.5, 5.0, 10.0] },
    'sum_mode': { 'values' : ['node', 'both'] },
    'lr':  { 'values' : [0.00005, 0.0001, 0.0005, 0.001] },
    'weight_decay' : { 'values' : [1e-5, 1e-4, 1e-3, 0] },
    }


if local:
    parameters_dict.update({'graph_mode': { 'value': 'vector_masked' }})
else:
    parameters_dict.update({'graph_mode': { 'values': ['energy', 'vector'] }})

if script_args.equivariant:
    parameters_dict.update({'n_v': { 'values': [16, 32, 48, 64] }})
    parameters_dict.update({ 'invariant': { 'value': False} })
else:
    parameters_dict.update({'n_v': { 'value': None }})
    parameters_dict.update({ 'invariant': { 'value': True} })

parameters_dict.update({ 'subset': { 'value': None} })
parameters_dict.update({ 'dataset': { 'value': dataset_full[dataset]} })
parameters_dict.update({ 'num_epochs': { 'value': epochs[dataset]} })
parameters_dict.update({ 'train_frac': { 'value': train_frac[dataset]} })
parameters_dict.update({ 'noH': { 'value': True} })
parameters_dict.update({ 'geometry': { 'value': geometry[dataset]} })
parameters_dict.update({ 'random_baseline': { 'value': False} })
parameters_dict.update({ 'features': { 'value': features[dataset]} })
parameters_dict.update({ 'target_column': { 'value': target_column} })
parameters_dict.update({ 'seed': { 'value': 123} })
parameters_dict.update({ 'splitter': { 'value': splitter[dataset]} })
parameters_dict.update({ 'classification': { 'value': False }})
parameters_dict.update({ 'arch': { 'value': 'normal' }})

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

wandb_name = 'test'
sweep_id = wandb.sweep(sweep_config, project=project) if script_args.id is None else script_args.id
print(sweep_id)
wandb.agent(sweep_id, train_wrapper, count=script_args.num, project=project)
