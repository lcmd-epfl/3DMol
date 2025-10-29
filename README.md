```
conda install python=3.10.10
pip install scipy numpy
conda config --add channels pyg
conda config --add channels nvidia
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit==10.2.8 -c pytorch -c nvidia
conda install networkx==2.8.4 h5py==3.7
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install e3nn
pip install rdkit==2023.03.1
pip install pyaml wandb ase
conda install pyg
conda install pandas==2.0.2
pip install chemprop==1.6.1
pip install git+https://github.com/lcmd-epfl/cell2mol
pip install morfeus-ml==0.7.2
```
