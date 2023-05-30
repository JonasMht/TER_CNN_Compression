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
session_name = "final_data_collection"

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
		


# Load data
dataset_folder_i3 = "/home/mehtali/TER_CNN_Compression/Data/training-data/data/IGBMC_I3_diversifie/patches/"
validate_list_i3 = dataset_folder_i3+"test_1000.txt"

# TODO
dataset_folder_lw4 = "/home/mehtali/TER_CNN_Compression/Data/training-data/data/IGBMC_LW4_diversifie/patches/"
validate_list_lw4 = dataset_folder_lw4+"test_5000-7500_lw4.txt"


validate_dataset_i3 = SegmentationDataSet(root=dataset_folder_i3,
								list_path=validate_list_i3
								)

validate_dataset_lw4 = SegmentationDataSet(root=dataset_folder_lw4,
								list_path=validate_list_lw4
								)


validation_dataloader_i3 = torch.utils.data.DataLoader(validate_dataset_i3,
											  batch_size=batch_size,
											  shuffle=True,
											  num_workers=workers)

validation_dataloader_lw4 = torch.utils.data.DataLoader(validate_dataset_lw4,
											  batch_size=batch_size,
											  shuffle=True,
											  num_workers=workers)



"""
# Example
teacher_path = get_first_file_path( get_folder_paths_like(model_path, "teacher")[0])
print(teacher_path)
teacher_model = load_model(UNet_modular(channel_depth=32, n_channels=3, n_classes=1), device, teacher_path)
"""

if False:
	# Draw the input ground trouth and the output of all the models in one line
	# Load the models
	teacher_model_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/teacher_model_depth_128/teacher_model_param_53553921.pth"
	teacher_model = load_model(UNet_modular(channel_depth=128, n_channels=3, n_classes=1), device, teacher_model_path)

	d_model_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/d_model_depth_3/iter_2/d_model_depth_3_iteration_2_param_30046.pth"
	nd_mode_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/nd_model_depth_3/iter_3/nd_model_depth_3_iteration_3_param_30046.pth"

	d_model = load_model(UNet_modular(channel_depth=3, n_channels=3, n_classes=1), device, d_model_path)
	nd_model = load_model(UNet_modular(channel_depth=3, n_channels=3, n_classes=1), device, nd_mode_path)

	# Get a batch of images
	input, target = next(iter(validation_dataloader_i3))

	# Get the output images
	teacher_output = teacher_model(input)
	d_output = d_model(input)
	nd_output = nd_model(input)

	# Clamp the output to 0-1
	teacher_output = torch.clamp(teacher_output, 0, 1)
	d_output = torch.clamp(d_output, 0, 1)
	nd_output = torch.clamp(nd_output, 0, 1)

	# Draw the images on one line having input, ground trouth, teacher output, d output, nd output
	# Draw the images
	fig, axs = plt.subplots(1, 5, figsize=(15, 5))
	axs[0].imshow(input[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[0].set_title("Input")
	axs[1].imshow(target[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[1].set_title("Ground trouth")
	axs[2].imshow(teacher_output[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[2].set_title("Teacher CNN output")
	axs[3].imshow(d_output[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[3].set_title("Distilled CNN output")
	axs[4].imshow(nd_output[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[4].set_title("Not distilled CNN output")
	# Save the figure
	fig.savefig(fig_path+"results_i3.png")

	# On lw4
	# Get a batch of images
	input, target = next(iter(validation_dataloader_lw4))

	# Get the output images
	teacher_output = teacher_model(input)
	
	d_output = d_model(input)
	
	nd_output = nd_model(input)

	# Clamp the output to 0-1
	teacher_output = torch.clamp(teacher_output, 0, 1)
	d_output = torch.clamp(d_output, 0, 1)
	nd_output = torch.clamp(nd_output, 0, 1)

	# Draw the images on one line having input, ground trouth, teacher output, d output, nd output
	# Draw the images
	fig, axs = plt.subplots(1, 5, figsize=(15, 5))
	axs[0].imshow(input[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[0].set_title("Input")
	axs[1].imshow(target[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[1].set_title("Ground trouth")
	axs[2].imshow(teacher_output[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[2].set_title("Teacher CNN output")
	axs[3].imshow(d_output[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[3].set_title("Distilled CNN output")
	axs[4].imshow(nd_output[0,0,:,:].cpu().detach().numpy(), cmap='gray')
	axs[4].set_title("Not distilled CNN output")
	# Save the figure
	fig.savefig(fig_path+"results_lw4.png")



# The same but plotting an entire batch
if True:
	# Draw the input ground trouth and the output of all the models in one line
	# Load the models
	teacher_model_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/teacher_model_depth_128/teacher_model_param_53553921.pth"
	teacher_model = load_model(UNet_modular(channel_depth=128, n_channels=3, n_classes=1), device, teacher_model_path)

	d_model_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/d_model_depth_3/iter_2/d_model_depth_3_iteration_2_param_30046.pth"
	nd_mode_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/nd_model_depth_3/iter_3/nd_model_depth_3_iteration_3_param_30046.pth"

	d_model = load_model(UNet_modular(channel_depth=3, n_channels=3, n_classes=1), device, d_model_path)
	nd_model = load_model(UNet_modular(channel_depth=3, n_channels=3, n_classes=1), device, nd_mode_path)

	# Get a batch of images
	input, target = next(iter(validation_dataloader_i3))

	# Get the output images
	teacher_output = teacher_model(input)
	d_output = d_model(input)
	nd_output = nd_model(input)

	# Clamp the output to 0-1
	teacher_output = torch.clamp(teacher_output, 0, 1)
	d_output = torch.clamp(d_output, 0, 1)
	nd_output = torch.clamp(nd_output, 0, 1)

	# Plot the entire batch with one line per image having input, ground trouth, teacher output, d output, nd output
	wanted_batch_size = 5
	fig, axs = plt.subplots(wanted_batch_size, 5, figsize=(15, 25))
	axs[0,0].set_title("Input")
	axs[0,1].set_title("Ground trouth")
	axs[0,2].set_title("Teacher CNN output")
	axs[0,3].set_title("Distilled CNN output")
	axs[0,4].set_title("Not distilled CNN output")
	for i in range(wanted_batch_size):
		axs[i,0].imshow(input[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,1].imshow(target[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,2].imshow(teacher_output[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,3].imshow(d_output[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,4].imshow(nd_output[i,0,:,:].cpu().detach().numpy(), cmap='gray')
	
	# Save the figure
	fig.savefig(fig_path+"results_i3_batch.png")

	# On lw4
	# Get a batch of images
	input, target = next(iter(validation_dataloader_lw4))

	# Get the output images
	teacher_output = teacher_model(input)
	d_output = d_model(input)
	nd_output = nd_model(input)
	
	# Clamp the output to 0-1
	teacher_output = torch.clamp(teacher_output, 0, 1)
	d_output = torch.clamp(d_output, 0, 1)
	nd_output = torch.clamp(nd_output, 0, 1)

	# Plot the entire batch with one line per image having input, ground trouth, teacher output, d output, nd output
	fig, axs = plt.subplots(batch_size, 5, figsize=(15, 25))
	axs[0,0].set_title("Input")
	axs[0,1].set_title("Ground trouth")
	axs[0,2].set_title("Teacher CNN output")
	axs[0,3].set_title("Distilled CNN output")
	axs[0,4].set_title("Not distilled CNN output")
	for i in range(batch_size):
		axs[i,0].imshow(input[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,1].imshow(target[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,2].imshow(teacher_output[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,3].imshow(d_output[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		axs[i,4].imshow(nd_output[i,0,:,:].cpu().detach().numpy(), cmap='gray')
		
	
	# Save the figure
	fig.savefig(fig_path+"results_lw4_batch.png")



if False:
	
	# Test the models here and save their performance as csv
	# Data collection

	# Load all the models
	# Load the teacher model

	teacher_model_path = "/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/teacher_model_depth_128/teacher_model_param_53553921.pth"
	teacher_model = load_model(UNet_modular(channel_depth=128, n_channels=3, n_classes=1), device, teacher_model_path)




	d_model_paths = [
		"/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/d_model_depth_3/iter_1/d_model_depth_3_iteration_1_param_30046.pth",
		"/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/d_model_depth_3/iter_2/d_model_depth_3_iteration_2_param_30046.pth",
		"/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/d_model_depth_3/iter_3/d_model_depth_3_iteration_3_param_30046.pth",
	]

	nd_model_paths = [
		"/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/nd_model_depth_3/iter_1/nd_model_depth_3_iteration_1_param_30046.pth",
		"/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/nd_model_depth_3/iter_2/nd_model_depth_3_iteration_2_param_30046.pth",
		"/home/mehtali/TER_CNN_Compression/Data/Saves/final_data_collection/network_weigths/nd_model_depth_3/iter_3/nd_model_depth_3_iteration_3_param_30046.pth",
	]


	# Create csv file
	csv_file = open(log_path+"test_performance.csv", "w")

	# Write the header
	csv_file.write("model_name, parameters, training, i3_dice, lw4_dice\n")
	csv_file.flush()

	# Write the teacher model performance
	i3_dice = evaluate(teacher_model, validation_dataloader_i3)
	lw4_dice = evaluate(teacher_model, validation_dataloader_lw4)
	csv_file.write("teacher_model, {}, {}, {}, {}\n".format(get_trainable_param(teacher_model), "teacher", i3_dice, lw4_dice))
	csv_file.flush()

	# Write the distilled models performance
	for i in range(len(d_model_paths)):
		d_model_path = d_model_paths[i]
		d_model = load_model(UNet_modular(channel_depth=3, n_channels=3, n_classes=1), device, d_model_path)
		i3_dice = evaluate(d_model, validation_dataloader_i3)
		lw4_dice = evaluate(d_model, validation_dataloader_lw4)
		csv_file.write("d_model_iter_{}, {}, {}, {}, {}\n".format(i+1, get_trainable_param(d_model), "d_model", i3_dice, lw4_dice))
		csv_file.flush()
	
	# Write the non distilled models performance
	for i in range(len(nd_model_paths)):
		nd_model_path = nd_model_paths[i]
		nd_model = load_model(UNet_modular(channel_depth=3, n_channels=3, n_classes=1), device, nd_model_path)
		i3_dice = evaluate(nd_model, validation_dataloader_i3)
		lw4_dice = evaluate(nd_model, validation_dataloader_lw4)
		csv_file.write("nd_model_iter_{}, {}, {}, {}, {}\n".format(i+1, get_trainable_param(nd_model), "nd_model", i3_dice, lw4_dice))
		csv_file.flush()

	csv_file.close()


