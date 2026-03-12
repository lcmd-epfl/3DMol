# 3DMol

Geometric deep learning model for molecular properties based on [e3nn](https://e3nn.org)
applied to transition metal complexes.

Preprint: https://chemrxiv.org/doi/10.26434/chemrxiv-2025-j38bv

## Installation

Change torch and cuda versions to the ones that work for your hardware and run:
```
TORCH="2.7.0"
CUDA="126"
ENV_NAME="3dmol"
conda create -n ${ENV_NAME} python=3.12.11
conda activate ${ENV_NAME}
conda install numpy==1.26.4
conda install tqdm
conda install networkx==3.5 h5py==3.14.0 pandas==2.3.1
pip install torch==${TORCH} torchvision torchaudio
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-${TORCH}+cu${CUDA}.html
pip install pyaml==25.7.0 wandb==0.21.0 ase==3.25.0
pip install e3nn==0.5.1
pip install rdkit==2025.3.3
pip install morfeus-ml==0.7.2
pip install git+https://github.com/lcmd-epfl/cell2mol
```

## Usage

The repository contains two main scripts:
* `train.py` for model training and test set inference and evaluation,
* `sweep.py` for hyperparameter optimization.

### Hyperparameter optimization

The sweep script uses [W&B's](https://wandb.ai/) optimization algorithm, so to run it one needs an account.
Our results are here: [https://wandb.ai/equireact/3dmol-TMC-benchmark/sweeps](https://wandb.ai/equireact/3dmol-TMC-benchmark/sweeps).

For example, the following command runs a sweep
for spin-splitting energy on TMGSspinPlus
using the local (metal atom only) representation:
```
wandb enabled
./sweep.py --dataset TMGSspinPlus --target splitting --local
```
See also a submission script [example](data/TMGSspinPlus/results/0_submit-sweep.sbatch).

### Training

For training and evaluation the logging with W&B is optional.
Our results are here: [https://wandb.ai/equireact/3dmol-TMC-benchmark/](https://wandb.ai/equireact/3dmol-TMC-benchmark/).


For example, the following command trains an invariant (`--invariant`) model
for spin-splitting energy (`--target_column splitting`)
on TMGSspinPlus (`--dataset data/TMGSspinPlus/dataloader_TMGSspinPlus.py:TMGSspinPlus`)
using the local (metal atom only) representation (`--graph_mode vector_masked`).
and excluding H atoms from the molecular graphs (`--no-H`):
```
wandb enabled

./train.py \
    --experiment_name 3DMol-TMGSspinPlus-cv \
    --wandb_name cv10-TMGSspinPlus-splitting-inv-noH-ns64-d64-l2-vector_masked-node \
    --logdir ./ \
    --device cuda \
    --num_epochs 512 \
    --seed 123 \
    --print_predictions \
    \
    --dataset data/TMGSspinPlus/dataloader_TMGSspinPlus.py:TMGSspinPlus \
    --splitter test:data/TMGSspinPlus/splits/0_test_indices.txt \
    --target_column splitting \
    \
    --invariant  \
    --noH  \
    --graph_mode vector_masked \
    \
    --distance_emb_dim 64 \
    --dropout_p 0.05 \
    --lr 0.001 \
    --max_neighbors 25 \
    --n_conv_layers 2 \
    --n_s 64 \
    --n_v 0 \
    --radius 5.0 \
    --sum_mode node \
    --train_frac 0.8 \
    --weight_decay 1e-05 \
```
The results (log and checkpoint files) will be saved to `%LOGDIR/%EXPERIMENT_NAME/%WANDB_NAME.*`.
We recommend to set `--logdir` to your scratch directory.

All the submission scripts used can be found in `data/*/results/sbatch/*.sbatch`.


## Data

The `data` dicrectory contains three subdirectories for the datasets studied in the paper:
* [TM-GSspin⁺](data/TMGSspinPlus),
* [tmPHOTO](data/tmPHOTO),
* [Octa-MK](data/OctaMK).

Each contains
* a CSV file with properties (`*.csv`),
* a dataloader class (`dataloader_*.py`),
* a directory with test set indices for the splits for CV (`splits/`),
* `results/`. This directory contains submission scripts (`sbatch/`),
  sweep (`sweep_best_runs.csv`) and runs (`cv_runs.csv`, `results.csv`),
  along with the scripts generating them using W&B's API.
