# Note
Light weight  Crack segmentation model
Need Linux system

Open:https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0
Donwload:causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# PIP and Check weather installed
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install mamba

source /etc/network_turbo  
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e . --no-build-isolation
conda list

cd mamba
git checkout v2.2.0  # Change to old version
pip install transformers==4.34.0
pip install -e . --no-build-isolation

#After Installed mamba you can install other pages

# Envariment
    - python ： 3.10
    - pytorch ：2.1.2
    - cuda  :  11.8

# Need to know

u can make your files like this：

models---
     CS4.py
     VSSmamba.py
     
dataset.py

engine---
    train.py

tools----
     profile_model.py

train2.ipynb

visualization.ipynb

heatmap.ipynb




# Need fully Version ： Contact： 1971777601@qq.com





# U can shuffle your datasets Like this  


datasets---
       train.txt
       val.txt
       test.txt




