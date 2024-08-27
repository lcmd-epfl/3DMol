import argparse
import pprint
import wandb
from train import train
import os


def train_wrapper():
    with wandb.init(config=None):
        args = wandb.config
        try:
            train(run_dir, logname, project, wandb_name, args,
                  device='cuda', num_epochs=args.num_epochs, checkpoint=None,
                  verbose=False, print_predictions=False, eval_on_test=False,
                  sweep=True, print_repr=False,
                  dataset=args.dataset, process=False, noH=args.noH,
                  xtb=args.xtb, xtb_subset=args.xtb_subset,
                  target_column=args.target_column, features=args.features,
                  splitter=args.splitter, subset=args.subset, training_fractions=[args.train_frac], CV=1)
        except Exception as e:
            print(e)
            pass



parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='yuri', help='dataset')
parser.add_argument('-t', '--target', default='gap', help='target column')
script_args = parser.parse_args()

features = {'yuri': 'torchchem_v1'}
epochs = {'yuri': 128}
target_columns_good = {'yuri': ('homo', 'lumo', 'gap', 'dipole_moment_debye', 'splitting', 'hirshfeld', 'n_fod')}
project = 'nequimol'

dataset = script_args.dataset
target_column = script_args.target
if target_column.lower() not in target_columns_good[dataset]:
    raise RuntimeError
run_dir = f'sweep_{dataset}_{target_column}'
if not os.path.exists(run_dir):
    try:
        os.makedirs(run_dir)
    except:
        pass
logname = 'sweep.log'

wandb.login()

metric = { 'name': 'val_score_best', 'goal': 'minimize' }
sweep_config = { 'method': 'bayes', 'metric': metric }

parameters_dict = {
    'distance_emb_dim': { 'values': [16, 32, 48, 64] },
    'dropout_p': { 'values': [0.0, 0.05, 0.1] },
    'graph_mode': { 'values': ['energy', 'vector'] },
    'max_neighbors': { 'values': [10, 25, 50] },
    'n_conv_layers': { 'values': [2, 3] },
    'n_s': { 'values': [16, 32, 48, 64] },
    'n_v': { 'values': [16, 32, 48, 64] },
    'radius': { 'values' : [2.5, 5.0, 10.0] },
    'sum_mode': { 'values' : ['node', 'both'] },
    'lr':  { 'values' : [0.00005, 0.0001, 0.0005, 0.001] },
    'weight_decay' : { 'values' : [1e-5, 1e-4, 1e-3, 0] },
    }

parameters_dict.update({ 'invariant': { 'value': True} })
parameters_dict.update({ 'subset': { 'value': None} })
parameters_dict.update({ 'dataset': { 'value': dataset} })
parameters_dict.update({ 'num_epochs': { 'value': epochs[dataset]} })
parameters_dict.update({ 'train_frac': { 'value': 0.8} })
parameters_dict.update({ 'noH': { 'value': True} })
parameters_dict.update({ 'two_layers_atom_diff': { 'value': False} })
parameters_dict.update({ 'xtb': { 'value': False} })
parameters_dict.update({ 'xtb_subset': { 'value': False} })
parameters_dict.update({ 'random_baseline': { 'value': False} })
parameters_dict.update({ 'features': { 'value': features[dataset]} })
parameters_dict.update({ 'target_column': { 'value': target_column} })
parameters_dict.update({ 'seed': { 'value': 123} })
parameters_dict.update({ 'splitter': { 'value': 'random'} })

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

wandb_name = 'test'
sweep_id = wandb.sweep(sweep_config, project=project)
print(sweep_id)
wandb.agent(sweep_id, train_wrapper, count=64, project=project)
