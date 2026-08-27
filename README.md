```
TORCH="2.10.0"
CUDA="128"
ENV_NAME="3dmol"
conda create -n ${ENV_NAME} python=3.12.11
conda activate ${ENV_NAME}
conda install numpy==2.5.2
conda install tqdm
conda install networkx==3.5 h5py==3.14.0 pandas==2.3.1
pip install torch==${TORCH}
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_geometric -f https://data.pyg.org/whl/torch-${TORCH}+cu${CUDA}.html
pip install pyaml wandb ase
pip install e3nn==0.5.1
pip install rdkit==2025.3.3
pip install morfeus-ml==0.7.2
pip install git+https://github.com/lcmd-epfl/cell2mol
```
