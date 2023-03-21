#!/usr/bin/bash

device=$(1:="gpu")

conda create -n swmp python=3.8 -y

# use 'source' instead of 'conda' in bash env
source activate swmp

# conda install env
if [ "$device" == "cpu"]
then 
  echo "Install packages with CPU only"
  conda install pytorch torchvision torchaudio cpuonly pyg tensorboard \
    matplotlib seaborn joblib networkx numba \
    ipykernel flake8 autopep8 graphviz jupyter ipywidgets pytest\
    -c pytorch -c pyg -y
elif [ "$device" == "gpu" ]
  echo "Install packages with CUDA available"
  conda install pytorch torchvision torchaudio cudatoolkit=11.6 pyg tensorboard \
    matplotlib seaborn joblib networkx numba \
    ipykernel flake8 autopep8 graphviz jupyter ipywidgets pytest \
    -c pytorch -c nvidia -c pyg 
else
  echo "Please choose from 'cpu' oro 'gpu'."
fi

pip install pandapower pygraphviz deephyper

