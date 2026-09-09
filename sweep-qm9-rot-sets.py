import os
import sys
from types import SimpleNamespace
from datetime import datetime
import argparse
import pprint
import wandb
from train import train, Logger


def train_wrapper():
    with wandb.init(config=None):
        args_dict = args_table[wandb.config['from_sweep']]
        args = SimpleNamespace(**args_dict)
        pprint.pprint(args)
        try:
            train(run_dir, logname, script_args.project, wandb_name, args_dict,
                  device='cuda', num_epochs=args.num_epochs, checkpoint=None,
                  verbose=False, print_predictions=False, eval_on_test=False,
                  sweep=True, print_repr=False,
                  dataset=args.dataset, process=False, noH=args.noH,
                  geometry=args.geometry,
                  target_column=args.target_column, features=args.features,
                  splitter=args.splitter, subset=args.subset, training_fractions=[args.train_frac], CV=1)
        except Exception as e:
            print(e)


parser = argparse.ArgumentParser()
parser.add_argument('--project', default='3dmol-rot', help='project-name')
parser.add_argument('-d', '--dataset', default='qm9-rotation', help='dataset')
parser.add_argument('-t', '--target', default='rot589', help='target column')
parser.add_argument('--id', default=None, help='sweep id if continue a sweep')
parser.add_argument('-n', '--num', default=32, help='number of runs')
parser.add_argument('--arch', default='normal', help='arch')
parser.add_argument('--seed', default=666, help='seed')
script_args = parser.parse_args()

features = {
            'qm9-rotation': 'torchchem_v1',
            }
geometry = {
        'qm9-rotation': 'dft',
        }
epochs = {
        'qm9-rotation': 16,
        }
target_columns_good = {
        'qm9-rotation': ('rot589', 'rot633', 'rot355', 'rot589_sign', 'rot633_sign', 'rot355_sign', 'rot589_abs', 'rot633_abs', 'rot355_abs'),
        }
splitter = {
        'qm9-rotation': "test:data/qm9-rotation/splits/test.0.dat;val:data/qm9-rotation/splits/val.0.dat",
        }
dataset_full = {
        'qm9-rotation': 'data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation',
        }
train_frac = {
    'qm9-rotation': 0.8,
    }

dataset = script_args.dataset
target_column = script_args.target
if target_column not in target_columns_good[dataset]:
    raise RuntimeError
run_dir = f'sweep_{dataset}_{target_column}_{datetime.now().strftime("%y%m%d-%H%M%S.%f")}'
if not os.path.exists(run_dir):
    try:
        os.makedirs(run_dir)
    except:
        pass


classification_targets = ('rot589_sign', 'rot633_sign', 'rot355_sign')
classification = target_column in classification_targets

logname = 'sweep'
logpath = os.path.join(run_dir, f'{logname}.log')
sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

wandb.login()

metric = {'name': 'val_score_best', 'goal': ('maximize' if classification else 'minimize')}
sweep_config = { 'method': 'grid', 'metric': metric, 'name': f'{target_column}_{script_args.arch}_old-best' }

old_values = {
'k2ln8dnk':    [48,  0.05  ,  0.0005 , 48 , 48 , 5  , 0],
'fz0dajg7':    [64,  0.1   ,  0.0005 , 48 , 48 , 10 , 0],
'q3r6sjjn':    [32,  0     ,  0.0005 , 48 , 48 , 10 , 0],
'2i1u7frz':    [48,  0.05  ,  0.0001 , 48 , 48 , 10 , 0],
'umtzjghf':    [48,  0     ,  0.0005 , 48 , 48 , 5  , 0],
'grb0q42j':    [64,  0     ,  0.0005 , 48 , 32 , 2.5, 0],
'ins8adml':    [64,  0     ,  0.0005 , 48 , 32 , 5  , 0],
'kjuvm99h':    [64,  0.05  ,  0.0005 , 48 , 32 , 5  , 0],
'p3mh45nv':    [32,  0     ,  0.001  , 48 , 32 , 10 , 0],
'5ybavebt':    [64,  0.05  ,  0.0005 , 48 , 32 , 5  , 0],
'x1bxjt12':    [64,  0     ,  0.001  , 32 , 48 , 5  , 0],
'8za5pkjr':    [48,  0     ,  0.0005 , 32 , 32 , 5  , 0],
'5k9ptrkn':    [16,  0     ,  0.0005 , 16 , 32 , 5  , 0],
}
old_keys = ['distance_emb_dim', 'dropout_p',    'lr',   'n_s',  'n_v',  'radius',   'weight_decay']

constants = {
        'invariant': False,
        'classification': classification,
        'arch':            script_args.arch,
        'n_conv_layers': 3,
        'graph_mode': 'vector',
        'subset': None,
        'dataset': dataset_full[dataset],
        'num_epochs': epochs[dataset],
        'train_frac': train_frac[dataset],
        'noH': False,
        'geometry': geometry[dataset],
        'features': features[dataset],
        'target_column': target_column,
        'seed': script_args.seed,
        'splitter': splitter[dataset],
        'internal_weights': False,
        'optimizer': 'AdamW',
        }

args_table = {key: dict(zip(old_keys, val)) | constants for key, val in old_values.items()}

sweep_config_dict = {'from_sweep': {'values': sorted(list(args_table.keys()))  }  }
sweep_config['parameters'] = sweep_config_dict
pprint.pprint(sweep_config)

wandb_name = 'test'
sweep_id = wandb.sweep(sweep_config, project=script_args.project) if script_args.id is None else script_args.id
print(sweep_id)
wandb.agent(sweep_id, train_wrapper, count=script_args.num, project=script_args.project)
