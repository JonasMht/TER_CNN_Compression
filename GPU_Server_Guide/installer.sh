#!/bin/bash

virtualenv -p python3 env
source env/bin/activate 
pip install numpy
pip install scipy
pip install torch
pip install torchvision
pip install scikit-image
pip install scikit learn
pip install h5py

cat >> tmp_install.py << EOF
import torch
print("\n\n")
print("######################################################")
print("Is cuda avaible ? :")
if torch.cuda.is_available() :
    print("Yes !")
    print("You're now able to work with Pytorch on GPU's !")
    print("######################################################")
else :
    print("No :(")
    print("######################################################")
EOF

python tmp_install.py
rm tmp_install.py




