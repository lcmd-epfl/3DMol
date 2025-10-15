import os
from datetime import datetime
import argparse
import pprint
import wandb
from train import train


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
parser.add_argument('-d', '--dataset', default='qm9-rotation', help='dataset')
parser.add_argument('-t', '--target', default='rot589', help='target column')
parser.add_argument('--id', default=None, help='sweep id if continue a sweep')
parser.add_argument('-n', '--num', default=32, help='number of runs')
parser.add_argument('--arch', default='normal', help='arch')
parser.add_argument('--seed', default=666, help='arch')
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
        'qm9-rotation': ('rot589', 'rot633', 'rot355', 'rot589_sign', 'rot633_sign', 'rot355_sign', 'rot589_abs', 'rot633_abs', 'rot355_abs')
        }
splitter = {
        'qm9-rotation': 'random',
        }
dataset_full = {
        'qm9-rotation': 'data/qm9-rotation/dataloader_qm9-rotation.py:QM9Rotation',
        }
train_frac = {
    'qm9-rotation': 0.8,
    }
project = 'nequimol'

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
classification = True if target_column in classification_targets else False

logname = 'sweep.log'

wandb.login()

if classification:
    metric = { 'name': 'val_score_best', 'goal': 'maximize' }
else:
    metric = { 'name': 'val_score_best', 'goal': 'minimize' }
sweep_config = { 'method': 'bayes', 'metric': metric }

parameters_dict = {
    'distance_emb_dim': { 'values': [16, 32, 48, 64] },
    'dropout_p': { 'values': [0.0, 0.05, 0.1] },
    'max_neighbors': { 'values': [10, 25, 50] },
    'n_s': { 'values': [16, 32, 48] },
    'n_v': { 'values': [16, 32, 48] },
    'radius': { 'values' : [2.5, 5.0, 10.0] },
    'sum_mode': { 'values' : ['node', 'both'] },
    'lr':  { 'values' : [0.00005, 0.0001, 0.0005, 0.001] },
    }

parameters_dict.update({ 'weight_decay': { 'value': 0 }})
parameters_dict.update({ 'classification': { 'value': classification }})
parameters_dict.update({ 'invariant': { 'value': False }})
parameters_dict.update({ 'arch': { 'value': script_args.arch }})
parameters_dict.update({ 'n_conv_layers': { 'value': 3 }})
parameters_dict.update({ 'graph_mode': { 'value': 'vector' }})
parameters_dict.update({ 'subset': { 'value': None} })
parameters_dict.update({ 'dataset': { 'value': dataset_full[dataset]} })
parameters_dict.update({ 'num_epochs': { 'value': epochs[dataset]} })
parameters_dict.update({ 'train_frac': { 'value': train_frac[dataset]} })
parameters_dict.update({ 'noH': { 'value': False} })
parameters_dict.update({ 'geometry': { 'value': geometry[dataset]} })
parameters_dict.update({ 'random_baseline': { 'value': False} })
parameters_dict.update({ 'features': { 'value': features[dataset]} })
parameters_dict.update({ 'target_column': { 'value': target_column} })
parameters_dict.update({ 'seed': { 'value': script_args.seed } })
parameters_dict.update({ 'splitter': { 'value': splitter[dataset]} })
parameters_dict.update({ 'internal_weights': { 'value': False} })

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

wandb_name = 'test'
sweep_id = wandb.sweep(sweep_config, project=project) if script_args.id is None else script_args.id
print(sweep_id)
wandb.agent(sweep_id, train_wrapper, count=script_args.num, project=project)
