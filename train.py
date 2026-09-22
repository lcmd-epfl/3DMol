import os
import sys
import importlib.util
import argparse
from datetime import datetime
from getpass import getuser  # os.getlogin() won't work on a cluster
import random
from timeit import default_timer as timer
from collections import Counter
import faulthandler
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from trainer.metrics import MAE, Accuracy
from trainer.mol_trainer import MolTrainer
from models.equimol import EquiMol
from process.collate import CustomCollator
from process.splitter import split_dataset


faulthandler.enable()  # turn on for debugging for C code like Segmentation Faults
warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support")


class Logger:
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
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g_run = p.add_argument_group('external run parameters')
    g_run.add_argument('--experiment_name'    , type=str           , default=''       ,  help='name that will be added to the runs folder output')
    g_run.add_argument('--wandb_name'         , type=str           , default=None     ,  help='name of wandb run')
    g_run.add_argument('--project'            , type=str           , default='nequimol', help='name of wandb project')
    g_run.add_argument('--device'             , type=str           , default='cuda'   ,  help='device', choices=['cuda', 'cpu'])
    g_run.add_argument('--logdir'             , type=str           , default='logs'   ,  help='log dir')
    g_run.add_argument('--checkpoint'         , type=str           , default=None     ,  help='path of the checkpoint file to continue training')
    g_run.add_argument('--num_epochs'         , type=int           , default=2500     ,  help='number of times to iterate through all samples')
    g_run.add_argument('--patience'           , type=int           , default=150      ,  help='number of epochs with no improvement to stop early')
    g_run.add_argument('--gap_patience'       , type=int           , default=150      ,  help='number of epochs with gap > max. gap to stop early')
    g_run.add_argument('--max_gap'            , type=float         , default=None     ,  help='max. gap in std units between train and validation scores to stop early')
    g_run.add_argument('--verbose'            , action='store_true', default=False    ,  help='Print dims throughout the training process')
    g_run.add_argument('--process'            , action='store_true', default=False    ,  help='reprocess data')
    g_run.add_argument('--print_predictions'  , action='store_true', default=False    ,  help='print predictions for test molecules')
    g_run.add_argument('--print_repr'         , action='store_true', default=False    ,  help='print learned representations')
    g_run.add_argument('--print_atom_contrib' , action='store_true', default=False    ,  help='print atom contributions')
    g_run.add_argument('--fine_tuning'        , action='store_true', default=False    ,  help='if checkpoint is for fine-tuning')
    g_run.add_argument('--dataloader_args'    , type=str           , default=None     ,  help='additional dataloader arguments (key1:val1;key2:val2)')
    g_run.add_argument('--evaluation'         , action='store_true', default=False    ,  help='if evaluate on the full dataset')

    g_hyper = p.add_argument_group('hyperparameters')
    g_hyper.add_argument('--subset'               , type=int           , default=None           ,  help='size of a subset to use instead of the full set (tr+te+va)')
    g_hyper.add_argument('--n_s'                  , type=int           , default=48             ,  help='dimension of node features')
    g_hyper.add_argument('--n_v'                  , type=int           , default=48             ,  help='dimension of extra (p/d) features')
    g_hyper.add_argument('--n_conv_layers'        , type=int           , default=2              ,  help='number of conv layers')
    g_hyper.add_argument('--distance_emb_dim'     , type=int           , default=16             ,  help='how many gaussian funcs to use')
    g_hyper.add_argument('--radius'               , type=float         , default=5.0            ,  help='max radius of graph')
    g_hyper.add_argument('--dropout_p'            , type=float         , default=0.05           ,  help='dropout probability')
    g_hyper.add_argument('--seed'                 , type=int           , default=123            ,  help='seed')
    g_hyper.add_argument('--graph_mode'           , type=str           , default='vector'       ,  help='graph mode', choices=['vector', 'vector_masked'])
    g_hyper.add_argument('--dataset'              , type=str           ,                           help='dataset')
    g_hyper.add_argument('--splitter'             , type=str           , default='random'       ,  help='what splits to use: random / yasc / ydesc / test:path')
    g_hyper.add_argument('--noH'                  , action='store_true', default=False          ,  help='if remove H')
    g_hyper.add_argument('--invariant'            , action='store_true', default=False          ,  help='if use an invariant model')
    g_hyper.add_argument('--lr'                   , type=float         , default=0.001          ,  help='learning rate for adam')
    g_hyper.add_argument('--weight_decay'         , type=float         , default=0.0001         ,  help='weight decay for adam')
    g_hyper.add_argument('--train_frac'           , type=float         , default=0.8            ,  help='training fraction to use (val/te will be equally split over rest)')
    g_hyper.add_argument('--target_column'        , type=str           , default=None           ,  help='csv column with the target property')
    g_hyper.add_argument('--features'             , type=str           , default=None           ,  help='featurizer')
    g_hyper.add_argument('--geometry'             , type=str           , default=None           ,  help='geometry (dft/xtb/etc)')
    g_hyper.add_argument('--arch'                 , type=str           , default='normal'       ,  help='normal/both/pseudo')
    g_hyper.add_argument('--internal_weights'     , action='store_true', default=False          ,  help='if use internal weights in tensor products')
    g_hyper.add_argument('--classification'       , action='store_true', default=False          ,  help='if classification')
    g_hyper.add_argument('--batch_size'           , type=int           , default=8              ,  help='batch size')
    g_hyper.add_argument('--optimizer'            , type=str           , default='AdamW'        ,  help='optimizer', choices=['Adam', 'AdamW'])

    args = p.parse_args(arglist)

    arg_groups={}
    for group in p._action_groups:
        group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return args, arg_groups


def print_test_predictions(data, test_indices, targ_raw, pred_raw, *, classification=False):
    targ_raw = np.ravel(torch.vstack(targ_raw).cpu().numpy())
    pred_raw = np.ravel(torch.vstack(pred_raw).cpu().numpy())
    if classification:

        def sigmoid(x):
            return 1.0/(np.exp(-x)+1.0)

        def remap(x):
            return x*2.0-1.0

        def check_div(a, b):
            return a/b if b else np.inf

        targ_raw = remap(targ_raw)
        pred_raw = remap(sigmoid(pred_raw))
        targ = np.copy(targ_raw).astype(int)
        pred = np.zeros_like(pred_raw, dtype=int)
        pred[np.where(pred_raw<0)]=-1
        pred[np.where(pred_raw>=0)]=1
        err = pred-targ

        print('>>> # idx name target prediction_prob prediction error')
        for x in zip(test_indices, data.indices[test_indices], targ, pred_raw, pred, err, strict=True):
            print('>>>', *x, sep='\t')

        d_target = Counter(targ)
        d_pred = Counter(pred)
        d_err = Counter(err)

        N0 = d_target[-1]
        P0 = d_target[1]
        FN = d_err[-2]
        FP = d_err[2]
        N = d_pred[-1]
        P = d_pred[1]

        TN = (N0-FP)
        TP = (P0-FN)
        assert FP + TP == P
        assert FN + TN == N

        print(f'{TP=} {FN=} {FP=} {TN=}')
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        print(f'{accuracy=:.4f}')

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
        for x in zip(test_indices, data.indices[test_indices], targ_raw, pred_raw, targ, pred, pred-targ, strict=True):
            print('>>>', *x, sep='\t')


def evaluate_on_test(*, train_loader, val_loader, test_loader,
                     metrics, trainer, classification=False,
                     print_predictions=False, print_repr=False, print_atom_contrib=False):

    full_data = train_loader.dataset.dataset

    test_metrics, pred, targ = trainer.evaluation(test_loader, data_split='test', return_pred=True)

    if wandb.run is not None:
        std = full_data.std
        wandb.run.summary["test_score"] = test_metrics[metrics.main] * std
        if classification:
            wandb.run.summary["test_loss"] = test_metrics[metrics.loss_func_name]*std
        else:
            wandb.run.summary["test_rmse"] = np.sqrt(test_metrics[metrics.loss_func_name])*std

    if print_predictions:
        print_test_predictions(full_data, test_loader.dataset.indices, targ, pred, classification=classification)

    if print_repr or print_atom_contrib:
        for x_loader, x_title in [(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
            x_indices = x_loader.dataset.indices
            representations, atom_contrib = trainer.get_repr(x_loader, return_atom_contrib=print_atom_contrib)
            if print_repr:
                for x in zip(x_indices, representations, strict=True):
                    print(f'REPR>>> {x_title}', x[0], *x[1])
            if print_atom_contrib:
                for x in zip(x_indices, atom_contrib, strict=True):
                    print(f'ATOM_CONTRIB>>> {x_title}', x[0], *x[1])


def init_metrics(*, classification=False):
    if classification:
        return SimpleNamespace(
                all={'accuracy': Accuracy()},
                main='accuracy',
                main_goal='max',
                loss_func=BCEWithLogitsLoss(),
                loss_func_name='BCEWithLogitsLoss',
                )
    else:
        return SimpleNamespace(
                all={'mae': MAE()},
                main='mae',
                main_goal='min',
                loss_func=MSELoss(),
                loss_func_name='MSELoss',
                )


def init_dataloader(dataset):
    try:
        dataloader_path, dataloader_class = dataset.split(':')
        spec = importlib.util.spec_from_file_location('GenMolDataset', dataloader_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['GenMolDataset'] = mod
        spec.loader.exec_module(mod)
        MolDataloader = vars(mod)[dataloader_class]
        return MolDataloader
    except:
        raise NotImplementedError(f'Cannot load the {dataset} dataset.') from None


def train(run_dir, run_name, project, wandb_name, hyper_dict, *,
          device='cuda',
          num_epochs=512,
          checkpoint=False,
          fine_tuning=False,
          verbose=False,
          print_predictions=False,
          eval_on_test=True,
          sweep=False,
          print_repr=False,
          print_atom_contrib=False,
          patience=150,
          gap_patience=150,
          max_gap=None,
          evaluation=False,
          dataloader_args=None,
          process=False,
          # other (hidden)
          num_workers=0, pin_memory=False, val_per_batch=True, eval_per_epochs=0,
          minimum_epochs=0, models_to_save=None, clip_grad=100, log_iterations=100,
          lr_scheduler=ReduceLROnPlateau, factor=0.6, min_lr=8.0e-6, mode='max', lr_scheduler_patience=60,
          ):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Running on device {device}")

    seed = hyper_dict['seed']
    classification = hyper_dict['classification']
    batch_size = hyper_dict['batch_size']

    metrics = init_metrics(classification=classification)
    optim = {'Adam': Adam, 'AdamW': AdamW}[hyper_dict['optimizer']]

    dataloader_args_dict = None if dataloader_args is None else {f'_dl_extra_{key}': val for key, val in [entry.split(':') for entry in dataloader_args.split(';')]}

    MolDataloader = init_dataloader(hyper_dict['dataset'])

    time_start = timer()
    data = MolDataloader(process=process, classification=classification,
                         extra_args=dataloader_args_dict,
                         noH=hyper_dict['noH'], geometry=hyper_dict['geometry'],
                         target_column=hyper_dict['target_column'], graph_method=hyper_dict['features'])
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

    labels = data.labels.numpy()
    print(f"Data stdev {data.std:.4f}")
    print()

    if not sweep:
        wandb.init(project=project, config=hyper_dict, name=wandb_name, group=None)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = split_dataset(data, splitter=hyper_dict['splitter'], tr_frac=hyper_dict['train_frac'], subset=hyper_dict['subset'])

    print('MAE if use mean train for test:', (abs(labels[split.test]-(labels[split.train].mean())).mean()*data.std).item())

    print(f'total / train / test / val: {split.n} {len(split.train)} {len(split.test)} {len(split.val)}')
    train_data = Subset(data, split.train)
    val_data = Subset(data, split.val)
    test_data = Subset(data, split.test)

    model = EquiMol(node_fdim=data.input_node_feats_dim, verbose=verbose, device=device,
                    internal_weights=hyper_dict['internal_weights'],
                    arch=hyper_dict['arch'],
                    max_radius=hyper_dict['radius'],
                    n_s=hyper_dict['n_s'],
                    n_v=hyper_dict['n_v'],
                    n_conv_layers=hyper_dict['n_conv_layers'],
                    distance_emb_dim=hyper_dict['distance_emb_dim'],
                    graph_mode=hyper_dict['graph_mode'],
                    dropout_p=hyper_dict['dropout_p'],
                    invariant=hyper_dict['invariant'])

    print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    custom_collate = CustomCollator(device=device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                              pin_memory=pin_memory, num_workers=num_workers)

    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate, pin_memory=pin_memory,
                            num_workers=num_workers)

    trainer = MolTrainer(model=model, optim=optim, std=data.std, device=device,
                         loss_func=metrics.loss_func, metrics=metrics.all,
                         main_metric=metrics.main, main_metric_goal=metrics.main_goal,
                         run_dir=run_dir, run_name=run_name,
                         sampler=None, val_per_batch=val_per_batch,
                         checkpoint=checkpoint, fine_tuning=fine_tuning,
                         num_epochs=num_epochs,
                         eval_per_epochs=eval_per_epochs, patience=patience, gap_patience=gap_patience, max_gap=max_gap,
                         minimum_epochs=minimum_epochs, models_to_save=models_to_save,
                         clip_grad=clip_grad, log_iterations=log_iterations,
                         scheduler_step_per_batch=False,  # CHANGED THIS
                         lr=hyper_dict['lr'], weight_decay=hyper_dict['weight_decay'],
                         lr_scheduler=lr_scheduler, factor=factor, min_lr=min_lr, mode=mode,
                         lr_scheduler_patience=lr_scheduler_patience)

    time_start = timer()
    _val_metrics, _, _ = trainer.train(train_loader, val_loader)
    time_end = timer()
    print(f'\ntr_time: {time_end-time_start} s\n')

    if eval_on_test:
        time_start = timer()
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate,
                                 pin_memory=pin_memory, num_workers=num_workers)
        print('Evaluating on test, with test size: ', len(test_data))

        evaluate_on_test(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                         metrics=metrics, trainer=trainer, classification=classification,
                         print_predictions=print_predictions, print_repr=print_repr, print_atom_contrib=print_atom_contrib)

        time_end = timer()
        print(f'\nte_time: {time_end-time_start} s\n')

    if not sweep:
        wandb.finish()


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

    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "")
    logname = f'{args.wandb_name}-{SLURM_JOB_ID}-{datetime.now().strftime("%y%m%d-%H%M%S")}-{getuser()}'
    logpath = os.path.join(run_dir, f'{logname}.log')
    print(f"STDOUT> {logpath}")
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    project = args.project
    print(f'WANDB> {project}')
    print(f'WANDB> {args.wandb_name or "unspecified"}')

    print()
    print('COMMAND>', ' '.join(sys.argv))
    print()
    for key, val in vars(args).items():
        print(f'PARAMS> {key} : {val}')
    print()

    train(run_dir, logname, project, args.wandb_name, vars(arg_groups['hyperparameters']),
          device=args.device,
          num_epochs=args.num_epochs,
          checkpoint=args.checkpoint,
          fine_tuning=args.fine_tuning,
          verbose=args.verbose,
          print_predictions=args.print_predictions,
          eval_on_test=True,
          sweep=False,
          print_repr=args.print_repr,
          print_atom_contrib=args.print_atom_contrib,
          patience=args.patience,
          gap_patience=args.gap_patience,
          max_gap=args.max_gap,
          evaluation=args.evaluation,
          dataloader_args=args.dataloader_args,
          process=args.process,
          )
