# Name it whatever
conda create --name cython-torch
conda activate cython-torch

conda install -c anaconda cython -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge numpy -y
conda install pytorch torchvision cpuonly -c pytorch -y

# Windows 
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y