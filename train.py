import os
import sys
import argparse
from ast import literal_eval
import traceback
from datetime import datetime
from getpass import getuser  # os.getlogin() won't work on a cluster
import copy
import random
from collections import defaultdict
from timeit import default_timer as timer
import ast

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
import wandb

# turn on for debugging for C code like Segmentation Faults
import faulthandler
faulthandler.enable()

from trainer.metrics import MAE, Accuracy
from trainer.react_trainer import ReactTrainer
from models.equireact import EquiReact
from process.collate import CustomCollator
from process.splitter import split_dataset


class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def parse_arguments(arglist=sys.argv[1:]):
    p = argparse.ArgumentParser()

    g_run = p.add_argument_group('external run parameters')
    g_run.add_argument('--experiment_name'    , type=str           , default=''       ,  help='name that will be added to the runs folder output')
    g_run.add_argument('--wandb_name'         , type=str           , default=None     ,  help='name of wandb run')
    g_run.add_argument('--project'            , type=str           , default='nequimol', help='name of wandb project')
    g_run.add_argument('--device'             , type=str           , default='cuda'   ,  help='cuda or cpu')
    g_run.add_argument('--logdir'             , type=str           , default='logs'   ,  help='log dir')
    g_run.add_argument('--checkpoint'         , type=str           , default=None     ,  help='path of the checkpoint file to continue training')
    g_run.add_argument('--CV'                 , type=int           , default=1        ,  help='cross validate')
    g_run.add_argument('--num_epochs'         , type=int           , default=2500     ,  help='number of times to iterate through all samples')
    g_run.add_argument('--seed'               , type=int           , default=123      ,  help='initial seed values')
    g_run.add_argument('--verbose'            , action='store_true', default=False    ,  help='Print dims throughout the training process')
    g_run.add_argument('--process'            , action='store_true', default=False    ,  help='(re-)process data by force (if data is already there, default is to not reprocess)?')
    g_run.add_argument('--print_predictions'  , action='store_true', default=False    ,  help='print predictions for test molecules')
    g_run.add_argument('--print_repr'         , action='store_true', default=False    ,  help='print learned representations')
    g_run.add_argument('--learning_curve'     , action='store_true', default=False    ,  help='run learning curve (5 tr set sizes)')
    g_run.add_argument('--fine_tuning'        , action='store_true', default=False    ,  help='if checkpoint is for fine-tuning')
    g_run.add_argument('--dataloader_args'    , type=str           , default=None     ,  help='additional dataloader arguments (key1:val1;key2:val2)')

    g_hyper = p.add_argument_group('hyperparameters')
    g_hyper.add_argument('--subset'               , type=int           , default=None           ,  help='size of a subset to use instead of the full set (tr+te+va)')
    g_hyper.add_argument('--max_neighbors'        , type=int           , default=20             ,  help='max number of neighbors')
    g_hyper.add_argument('--n_s'                  , type=int           , default=48             ,  help='dimension of node features')
    g_hyper.add_argument('--n_v'                  , type=int           , default=48             ,  help='dimension of extra (p/d) features')
    g_hyper.add_argument('--n_conv_layers'        , type=int           , default=2              ,  help='number of conv layers')
    g_hyper.add_argument('--distance_emb_dim'     , type=int           , default=16             ,  help='how many gaussian funcs to use')
    g_hyper.add_argument('--radius'               , type=float         , default=5.0            ,  help='max radius of graph')
    g_hyper.add_argument('--dropout_p'            , type=float         , default=0.05           ,  help='dropout probability')
    g_hyper.add_argument('--sum_mode'             , type=str           , default='node'         ,  help='sum node (node, edge, or both)')
    g_hyper.add_argument('--graph_mode'           , type=str           , default='energy'       ,  help='prediction mode, energy, or vector')
    g_hyper.add_argument('--dataset'              , type=str           , default='cyclo'        ,  help='cyclo / gdb / proparg')
    g_hyper.add_argument('--splitter'             , type=str           , default='random'       ,  help='what splits to use: random / scaffold / yasc / ydesc / test:path')
    g_hyper.add_argument('--random_baseline'      , action='store_true', default=False          ,  help='random baseline (no graph conv)')
    g_hyper.add_argument('--noH'                  , action='store_true', default=False          ,  help='if remove H')
    g_hyper.add_argument('--invariant'            , action='store_true', default=False          ,  help='if run "InReact"')
    g_hyper.add_argument('--lr'                   , type=float         , default=0.001          ,  help='learning rate for adam')
    g_hyper.add_argument('--weight_decay'         , type=float         , default=0.0001         ,  help='weight decay for adam')
    g_hyper.add_argument('--train_frac'           , type=float         , default=0.9            ,  help='training fraction to use (val/te will be equally split over rest)')
    g_hyper.add_argument('--target_column'        , type=str           , default=None           ,  help='csv column with the target property')
    g_hyper.add_argument('--features'             , type=str           , default=None           ,  help='featurizer')
    g_hyper.add_argument('--geometry'             , type=str           , default=None           ,  help='geometry (dft/xtb/etc)')
    g_hyper.add_argument('--arch'                 , type=str           , default='normal'       ,  help='normal/both/pseudo')
    g_hyper.add_argument('--internal_weights'     , action='store_true', default=False          ,  help='if use internal weights in tensor products')
    g_hyper.add_argument('--classification'       , action='store_true', default=False          ,  help='if classification')
    g_hyper.add_argument('--embedding_specs'      , type=str           , default=None           ,  help='MACE extra embedding specs')

    args = p.parse_args(arglist)

    arg_groups={}
    for group in p._action_groups:
        group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    if args.CV > 1 and args.learning_curve:
        raise RuntimeError

    return args, arg_groups


def print_test_predictions(data, test_indices, targ_raw, pred_raw, classification=False):
    targ_raw = np.ravel(torch.vstack(targ_raw).cpu().numpy())
    pred_raw = np.ravel(torch.vstack(pred_raw).cpu().numpy())
    if classification:
        targ = np.copy(targ_raw).astype(int)
        pred = np.zeros_like(pred_raw, dtype=int)
        pred[np.where(pred_raw<0)]=-1
        pred[np.where(pred_raw>=0)]=1
        err = pred-targ

        print('>>> # idx name target prediction_raw prediction error')
        for x in zip(test_indices, data.indices[test_indices], targ, pred_raw, pred, err):
            print('>>>', *x, sep='\t')

        d_target = defaultdict(int, zip(*np.unique(targ, return_counts=True)))
        d_pred   = defaultdict(int, zip(*np.unique(pred, return_counts=True)))
        d_err    = defaultdict(int, zip(*np.unique(err, return_counts=True)))
        N0 = d_target[-1]
        P0 = d_target[1]
        FN = d_err[-2]
        FP = d_err[2]
        N = d_pred[-1]
        P = d_pred[1]

        TN = N0-FP
        TP = P0-FN
        assert FP + TP == P
        assert FN + TN == N

        print(f'{TP=} {FN=} {FP=} {TP=}')
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        print(f'{accuracy=:.4f}')

        def check_div(a, b):
            return a/b if b else np.inf

        recall = check_div(TP, TP+FN)
        FPR = check_div(FP, N0)
        precision = check_div(TP, P)
        F1 = check_div(TP, TP+(FP+FN)/2)
        print(f'+1: {recall=:.4f} {FPR=:.4f} {precision=:.4f} {F1=:.4f}')

        recall = check_div(TN, TN+FP)
        FNR = check_div(FN, P0)
        precision = check_div(TN, N)
        F1 = check_div(TN, TN+(FP+FN)/2)
        print(f'-1: {recall=:.4f} {FNR=:.4f} {precision=:.4f} {F1=:.4f}')


    else:
        data_std = data.std.item()
        data_mean = data.mean.item()
        targ = targ_raw*data_std+data_mean
        pred = pred_raw*data_std+data_mean
        print('>>> # idx name target_stdized prediction_stdized target prediction error')
        for x in zip(test_indices, data.indices[test_indices], targ_raw, pred_raw, targ, pred, pred-targ):
            print('>>>', *x, sep='\t')


def train(run_dir, run_name, project, wandb_name, hyper_dict,
          # run
          device='cuda',
          num_epochs=65536,
          checkpoint=False,
          fine_tuning=False,
          verbose=False,
          print_predictions=False,
          eval_on_test=True,
          sweep = False,
          print_repr=False,
          # dataset
          dataset=None,
          dataloader_args=None,
          process=False,
          noH=False,
          geometry=None,
          target_column=None,
          features=None,
          # splitting
          splitter='random',
          subset=None,
          training_fractions = [0.8],
          CV=0,
          seed0=123,
          # other (hidden)
          batch_size=8, num_workers=0, pin_memory=False,
          val_per_batch=True, eval_per_epochs=0, patience=150,
          minimum_epochs=0, models_to_save=[], clip_grad=100, log_iterations=100,
          lr_scheduler=ReduceLROnPlateau, factor=0.6, min_lr=8.0e-6, mode='max', lr_scheduler_patience=60,
          ):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Running on device {device}")

    classification = hyper_dict['classification']

    if classification:
        metrics = {'accuracy': Accuracy()}
        main_metric = 'accuracy'
        main_metric_goal = 'max'
        #loss_func = SoftMarginLoss()
        #loss_func_name = 'SoftMarginLoss'
        loss_func = MSELoss()
        loss_func_name = 'MSELoss'
    else:
        metrics = {'mae': MAE()}
        main_metric = 'mae'
        main_metric_goal = 'min'
        loss_func = MSELoss()
        loss_func_name = 'MSELoss'


    if hyper_dict['embedding_specs'] is None:
        embedding_specs = None
    else:
        embedding_specs = ast.literal_eval(hyper_dict['embedding_specs'])


    dataloader_args_dict = None if dataloader_args is None else {f'_dl_extra_{key}': val for  key, val in [entry.split(':') for entry in dataloader_args.split(';')]}

    if dataset=='proparg':
        from process.dataloader_proparg import PropargReactants as MolDataloader
    elif dataset=='test':
        from process.dataloader_test import TestSet as MolDataloader
    elif dataset=='dsC7O2H10nsd':
        from process.dataloader_qm9 import dsC7O2H10nsd as MolDataloader
    elif dataset=='qm9':
        from process.dataloader_qm9 import QM9 as MolDataloader
    else:
        try:
            dataloader_path, dataloader_class = dataset.split(':')
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location('GenMolDataset', dataloader_path)
            foo = importlib.util.module_from_spec(spec)
            sys.modules['GenMolDataset'] = foo
            spec.loader.exec_module(foo)
            MolDataloader = vars(foo)[dataloader_class]
        except:
            raise NotImplementedError(f'Cannot load the {dataset} dataset.')

    time_start = timer()
    data = MolDataloader(process=process, classification=classification,
                         extra_args=dataloader_args_dict,
                         noH=noH,
                         target_column=target_column, graph_method=features,
                         embedding_specs=embedding_specs)
    time_end = timer()
    print(f'\ndl_time: {time_end-time_start} s\n')

    print()
    for key, val in vars(data.parameters).items():
        print(f'PARAMS> {key} : {val}')
        hyper_dict[key] = val
    print()
    if dataloader_args_dict:
        for key, val in dataloader_args_dict.items():
            hyper_dict[key] = val
            print(f'PARAMS_DATALOADER> {key} : {val}')
        print()

    labels = data.labels
    std = data.std
    print(f"Data stdev {std:.4f}")
    print()

    for tr_frac in training_fractions:
        maes = []
        rmses = []
        seed = seed0

        for i in range(CV):
            print(f"CV iter {i+1}/{CV}")

            hyper_dict['CV iter'] = i
            hyper_dict['seed'] = seed
            if not sweep:
                if CV>1:
                    wandb.init(project=project, config=hyper_dict, name=f'{wandb_name}.cv{i}', group=wandb_name)
                    run_name_chk = f'{run_name}.cv{i}'
                elif len(training_fractions)>1:
                    hyper_dict['train_frac'] = f'{tr_frac}/{max(training_fractions)}'
                    print(f'{wandb_name}.tr{tr_frac}')
                    wandb.init(project=project, config=hyper_dict, name=f'{wandb_name}.tr{tr_frac}', group=None)
                    run_name_chk = f'{run_name}.tr{tr_frac}'
                else:
                    wandb.init(project=project, config=hyper_dict, name=wandb_name, group=None)
                    run_name_chk = run_name
            else:
                run_name_chk = run_name

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            tr_indices, te_indices, val_indices, indices = split_dataset(data, splitter=splitter,
                                                                         tr_frac=max(training_fractions),
                                                                         subset=subset)

            print('MAE if use mean train for test:', abs(labels.numpy()[te_indices]-(labels.numpy()[tr_indices].mean())).mean()*std)

            if len(training_fractions)>1:
                tr_indices = tr_indices[:int(tr_frac*len(indices))]


            print(f'total / train / test / val: {len(indices)} {len(tr_indices)} {len(te_indices)} {len(val_indices)}')
            train_data = Subset(data, tr_indices)
            val_data = Subset(data, val_indices)
            test_data = Subset(data, te_indices)

            # train sample
            label, idx, r0graph = train_data[0][:3]
            input_node_feats_dim = r0graph.x.shape[1]
            input_edge_feats_dim = 1
            if verbose:
                print(f"{r0graph=}")
                print(f"{input_node_feats_dim=}")
                print(f"{input_edge_feats_dim=}")

            model = EquiReact(node_fdim=input_node_feats_dim, edge_fdim=1, verbose=verbose, device=device,
                              embedding_specs = embedding_specs,
                              classification = hyper_dict['classification'],
                              internal_weights=hyper_dict['internal_weights'],
                              arch = hyper_dict['arch'],
                              max_radius=hyper_dict['radius'],
                              max_neighbors=hyper_dict['max_neighbors'],
                              sum_mode=hyper_dict['sum_mode'],
                              n_s=hyper_dict['n_s'],
                              n_v=hyper_dict['n_v'],
                              n_conv_layers=hyper_dict['n_conv_layers'],
                              distance_emb_dim=hyper_dict['distance_emb_dim'],
                              graph_mode=hyper_dict['graph_mode'],
                              dropout_p=hyper_dict['dropout_p'],
                              random_baseline=hyper_dict['random_baseline'],
                              invariant=hyper_dict['invariant'])
            print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

            sampler = None
            custom_collate = CustomCollator(device=device)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      pin_memory=pin_memory, num_workers=num_workers)

            val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate, pin_memory=pin_memory,
                                    num_workers=num_workers)


            trainer = ReactTrainer(model=model, std=std, device=device,
                                   metrics=metrics, loss_func=loss_func,
                                   main_metric=main_metric, main_metric_goal=main_metric_goal,
                                   run_dir=run_dir, run_name=run_name_chk,
                                   sampler=sampler, val_per_batch=val_per_batch,
                                   checkpoint=checkpoint, fine_tuning=fine_tuning,
                                   num_epochs=num_epochs,
                                   eval_per_epochs=eval_per_epochs, patience=patience,
                                   minimum_epochs=minimum_epochs, models_to_save=models_to_save,
                                   clip_grad=clip_grad, log_iterations=log_iterations,
                                   scheduler_step_per_batch = False, # CHANGED THIS
                                   lr=hyper_dict['lr'], weight_decay=hyper_dict['weight_decay'],
                                   lr_scheduler=lr_scheduler, factor=factor, min_lr=min_lr, mode=mode,
                                   lr_scheduler_patience=lr_scheduler_patience)

            time_start = timer()
            val_metrics, _, _ = trainer.train(train_loader, val_loader)
            time_end = timer()
            print(f'\ntr_time: {time_end-time_start} s\n')

            time_start = timer()
            if eval_on_test:
                test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate,
                                         pin_memory=pin_memory, num_workers=num_workers)
                print('Evaluating on test, with test size: ', len(test_data))

                # file dump for each split
                data_split_string = 'test_split_' + str(CV)
                test_metrics, pred, targ = trainer.evaluation(test_loader, data_split=data_split_string, return_pred=True)

                if print_predictions:
                    print_test_predictions(data, test_data.indices, targ, pred, classification=classification)

                if classification:
                    acc_split = test_metrics[main_metric] * std
                    loss_split = test_metrics[loss_func_name]*std
                    maes.append(acc_split)
                    rmses.append(loss_split)
                    if wandb.run is not None:
                        wandb.run.summary["test_score"] = acc_split
                        wandb.run.summary["test_loss"] = loss_split
                else:
                    mae_split = test_metrics[main_metric] * std
                    rmse_split = np.sqrt(test_metrics[loss_func_name])*std
                    maes.append(mae_split)
                    rmses.append(rmse_split)
                    if wandb.run is not None:
                        wandb.run.summary["test_score"] = mae_split
                        wandb.run.summary["test_rmse"] = rmse_split

                if print_repr:
                    for x_indices, x_loader, x_title in zip((train_data.indices, val_data.indices, test_data.indices),
                                                            (train_loader, val_loader, test_loader),
                                                            ('train', 'val', 'test')):
                        representations = trainer.get_repr(x_loader)
                        for x in zip(x_indices, representations):
                            print(f'>>>{x_title}', x[0], *x[1])

            time_end = timer()
            print(f'\nte_time: {time_end-time_start} s\n')

            seed += 1
            if not sweep:
                wandb.finish()

        if eval_on_test:
            if torch.__version__.split('.')[0]=='2':
                maes_  = torch.std_mean(torch.hstack(maes),  correction=0)
                rmses_ = torch.std_mean(torch.hstack(rmses), correction=0)
            else:
                maes_  = torch.std_mean(torch.hstack(maes), unbiased=False)
                rmses_ = torch.std_mean(torch.hstack(rmses), unbiased=False)
    return maes, rmses


if __name__ == '__main__':

    args, arg_groups = parse_arguments()

    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
    else:
        run_dir = os.path.join(args.logdir, args.experiment_name)
    if not os.path.exists(run_dir):
        print(f"creating run dir {run_dir}")
        try:
            os.makedirs(run_dir)
        except:
            pass

    SLURM_JOB_ID=os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else ""
    logname = f'{args.wandb_name}-{SLURM_JOB_ID}-{datetime.now().strftime("%y%m%d-%H%M%S")}-{getuser()}'
    logpath = os.path.join(run_dir, f'{logname}.log')
    print(f"STDOUT> {logpath}")
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    project = args.project
    print(f'WANDB> {project}')
    print(f'WANDB> {args.wandb_name if args.wandb_name else "unspecified"}')

    print()
    print('COMMAND>', ' '.join(sys.argv))
    print()
    for key, val in vars(args).items():
        print(f'PARAMS> {key} : {val}')
    print()

    if args.learning_curve:
        train_frac = args.train_frac * np.logspace(-4, 0, 5, endpoint=True, base=2)
    else:
        train_frac = [args.train_frac]
    print(f'TRAINING> {train_frac}')
    print()

    train(run_dir, logname, project, args.wandb_name, vars(arg_groups['hyperparameters']),
          # run
          device=args.device,
          num_epochs=args.num_epochs,
          checkpoint=args.checkpoint,
          fine_tuning=args.fine_tuning,
          verbose=args.verbose,
          print_predictions=args.print_predictions,
          eval_on_test=True,
          sweep=False,
          print_repr=args.print_repr,
          # dataset
          dataset=args.dataset,
          dataloader_args=args.dataloader_args,
          process=args.process,
          noH=args.noH,
          geometry=args.geometry,
          target_column=args.target_column,
          features=args.features,
          # splitting
          splitter=args.splitter,
          subset=args.subset,
          training_fractions=train_frac,
          CV=args.CV,
          seed0=args.seed,
          )
