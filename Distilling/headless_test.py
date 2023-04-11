from __future__ import print_function
# %matplotlib inline
import argparse
import os
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from IPython.display import HTML
from PIL import Image
from tqdm import tqdm
from torch import Tensor
import pickle
from datetime import datetime

import copy # To make a copy of a model


from torchvision import datasets, transforms
from torch.utils.data import random_split

from torchsummary import summary


import utils
from utils.dataset import *
from utils.model import *
from utils.model import UNet_modular
from utils.utils import *
import csv


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


"""
# If there are issues with the kernel, try the following :


python3 -m pip install ipykernel
python3 -m ipykernel install --user

python3 -m pip install ipykernel -U --user --force-reinstall

python3.6 -m pip install torchvision

# To run in the background use tmux

# Run:
~/env/bin/python headless_test.py 
"""

# Glbal teacher model

# Root directory for dataset
# Refaire nos data folder et tout pour que ce soit
# au format demandé par le dataloader

# Leave empty if you want to use the default name or load a previous session
session_name = "IGBMC_I3_2023-04-01_12:48:16"


dataset_folder = "/home/mehtali/TER_CNN_Compression/Data/training-data/data/IGBMC_I3_diversifie/patches/"

test_list = dataset_folder+"test_5000-7500_i3.txt"

# For debug
"""
test_list = dataset_folder+"test_20.txt"
"""


# Number of workers for dataloader
workers = 10

# Batch size during training (low batch_size if there are memory issues)
batch_size = 10

# Number of channels in the training images. For color images this is 3
nc = 3



# some net variable
amp = False

save_path =  "../Data/Saves/" + session_name+"/"
model_path = save_path+"network_weigths/"
log_path = save_path+"logs/"
fig_path = save_path+"fig/"


if not os.path.exists("../Data"):
	os.mkdir("../Data")
if not os.path.exists("../Data/Saves" ):
	os.mkdir("../Data/Saves" )

# We create this folder (only if it doesn't exists) to save weights of the training at some keys epoch
if not os.path.exists(save_path):
	os.mkdir(save_path)
	os.mkdir(model_path)
	os.mkdir(log_path)
	os.mkdir(fig_path)


# Decide which device we want to run on
device = torch.device("cuda:0" if (
	torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

# On regarde l'identifiant du GPU ou CPU sur lequel on travaille
print("device ID", device)
print("nom du GPU", torch.cuda.get_device_name(device))  # On vérifie son "nom"

torch.cuda.empty_cache()



# Not a good idea to augment the test data
test_dataset = SegmentationDataSet(root=dataset_folder,
								list_path=test_list
								)


test_dataloader = torch.utils.data.DataLoader(test_dataset,
											  batch_size=batch_size,
											  shuffle=True,
											  num_workers=workers)

def get_first_file_path(folder):
	"""
		Return the path of the first file in a folder
	"""
	for file in os.listdir(folder):
		return folder+'/'+file
	return None

def get_folder_paths(folder):
	"""
		Return the ordered paths of the folders in a folder
	"""
	folder_paths = []
	for file in os.listdir(folder):
		folder_paths.append(folder+file)
	return sorted(folder_paths)

def get_folder_paths_like(folder, pattern):
	"""
		Return the ordered paths of the folders in a folder which names contains the pattern
	"""
	folder_paths = []
	for folder_p in get_folder_paths(folder):
		if pattern in folder_p:
			folder_paths.append(folder_p+'/')
	return folder_paths

def get_models_like(folder, pattern):
	"""
		Return the ordered paths of the folders in a folder which names contains the pattern
	"""
	model_paths = []
	# Get all different models
	for model_type_path in get_folder_paths_like(folder, pattern):
		# get all variations of the model
		model_paths.append([])
		for model_variation_path in get_folder_paths(model_type_path):
			model_path = get_first_file_path(model_variation_path)
			if model_path is not None:
				model_paths[-1].append(model_path)
	return model_paths

def load_models(folder, pattern):
	"""
		Load a model from a list of paths
	"""
	model_paths = get_models_like(folder, pattern)
	models = []
	i = 1
	for model_type_path in model_paths:
		models.append([])
		for model_variation_path in model_type_path:
			print("Loading model", model_variation_path, "with channel depth", 2*i)
			model = UNet_modular(channel_depth=2*i, n_channels=3, n_classes=1)
			model = load_model(model, device, path=model_variation_path)
			models[-1].append(model)
		
		i+=1
		
	return models

"""
# Example
teacher_path = get_first_file_path( get_folder_paths_like(model_path, "teacher")[0])
print(teacher_path)
teacher_model = load_model(UNet_modular(channel_depth=32, n_channels=3, n_classes=1), device, teacher_path)
"""

nd_models = load_models(model_path, "/nd_model")

d_models = load_models(model_path, "/d_model")


# Create cvs file to save the evaluation results for each model type and each model variation

# Create the csv file
csv_file = open(log_path+"evaluation.csv", "w")
csv_file.write("model_type model_depth model_variation dice_coef\n")

i = 1
for depth_models in nd_models:
	j = 1
	for nd_model_variation in depth_models:
		dice = evaluate(nd_model_variation, test_dataloader)
		model_param = get_trainable_param(nd_model_variation)
		csv_file.write("nd_model "+str(model_param)+" "+str(j)+" "+str(dice).replace('.',',')+"\n")
		j+=1
	i+=1

i = 1
for depth_models in d_models:
	j = 1
	for d_model_variation in depth_models:
		dice = evaluate(d_model_variation, test_dataloader)
		model_param = get_trainable_param(d_model_variation)
		csv_file.write("d_model, "+str(model_param)+", "+str(j)+", "+str(dice)+"\n")
		j+=1
	i+=1

# Close the csv file
csv_file.close()



"""
# Load the teacher model
teacher_path = model_path +"teacher/"
teacher_model = load_model(UNet_modular(channel_depth=32, n_channels=3, n_classes=1), device)
"""