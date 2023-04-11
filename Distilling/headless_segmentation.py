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


# Run:
~/env/bin/python headless_segmentation.py 
"""

# Glbal teacher model
n_epochs = 150
n_models = 20
n_iter = 10


date = str(datetime.now()).split(' ')[0]
heure = str(datetime.now()).split(' ')[1].split('.')[0]

# Root directory for dataset
# Refaire nos data folder et tout pour que ce soit
# au format demandé par le dataloader

# Leave empty if you want to use the default name or load a previous session
session_name = ""

if len(session_name) == 0:
	session_name = "IGBMC_I3"+"_"+str(date)+"_"+str(heure)


dataset_folder = "/home/mehtali/TER_CNN_Compression/Data/training-data/data/IGBMC_I3_diversifie/patches/"

train_list = dataset_folder+"train_4000_i3.txt"
teacher_train_list = dataset_folder+"train_10000_i3.txt"
validate_list = dataset_folder+"test_1000.txt"

# For debug
"""
train_list = dataset_folder+"test_20.txt"
teacher_train_list = dataset_folder+"test_20.txt"
test_list = dataset_folder+"test_20.txt"
validate_list = dataset_folder+"test_20.txt"
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


log_file = open(log_path+"log.txt", "w")

log_file.write("dataset_folder :"+dataset_folder+"\n")
log_file.write("batch_size="+str(batch_size)+"\n")
log_file.write("num_max_epoch="+str(n_epochs)+"\n")
log_file.write("nc="+str(nc)+"\n")
log_file.close()

print("number of gpus :", torch.cuda.device_count())

# Decide which device we want to run on
device = torch.device("cuda:0" if (
	torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

# On regarde l'identifiant du GPU ou CPU sur lequel on travaille
print("device ID", device)
print("nom du GPU", torch.cuda.get_device_name(device))  # On vérifie son "nom"

torch.cuda.empty_cache()




# For data augmentation

geometric_augs = [
	# transforms.Resize((256, 256)), # Makes it easier to process using net
	# transforms.RandomRotation(degrees=(0, 180)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	# transforms.RandomRotation(45),
]

color_augs = [
	# transforms.ColorJitter(hue=0.05, saturation=0.4)
]


def make_tfs(augs):
	return transforms.Compose([transforms.ToPILImage()]+augs + [transforms.ToTensor()])


tfs = transforms.Compose(geometric_augs)


# Importation des images et masques de i3
train_dataset = SegmentationDataSet(root=dataset_folder,
							  list_path=train_list,
							  transform_img=make_tfs(
								  geometric_augs + color_augs),
							  transform_label=make_tfs(geometric_augs)
							  )

teacher_train_dataset = SegmentationDataSet(root=dataset_folder,
							  list_path=teacher_train_list,
							  transform_img=make_tfs(
								  geometric_augs + color_augs),
							  transform_label=make_tfs(geometric_augs)
							  )

# Not a good idea to augment the test data
validate_dataset = SegmentationDataSet(root=dataset_folder,
								list_path=validate_list
								)



train_dataloader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=batch_size,
											   shuffle=True,
											   num_workers=workers)

teacher_train_dataloader= torch.utils.data.DataLoader(teacher_train_dataset,
											   batch_size=batch_size,
											   shuffle=True,
											   num_workers=workers)

validation_dataloader = torch.utils.data.DataLoader(validate_dataset,
											  batch_size=batch_size,
											  shuffle=True,
											  num_workers=workers)


batch = next(iter(train_dataloader))

# On affiche quelques exemple du batch pour vérifier qu'on a bien importé les données
print("images source : ", batch[0].shape)
print("mask source :", batch[1].shape)





# Training

log_file = open(log_path+"model_performance.txt", "w")
log_file.write("Best Dice\tModel Name\n")
log_file.flush()

teacher_model_path = "/home/mehtali/TER_CNN_Compression/Distilling/teacher_model_param_3352257.pth"
# Teacher model
teacher_model = load_model(UNet_modular(channel_depth=32, n_channels=3, n_classes=1), device, teacher_model_path)

if teacher_model_path == "":
	model_descr = "teacher_model_param_{}".format(get_trainable_param(teacher_model))
	# Train teacher Model
	teacher_model, best_dice = model_training(teacher_model, teacher_train_dataloader, validation_dataloader, train, evaluate, n_epochs)
	teacher_model_path = model_path+"teacher/"
	if not os.path.exists(teacher_model_path):
		os.mkdir(teacher_model_path)
	save_model(teacher_model, teacher_model_path+model_descr+".pth")
	log_file.write("Dice : {:.3f}\tModel : {}\n".format(best_dice, model_descr))
	log_file.flush()

# Train smaller models
for i in range(1, n_models+1):
	d_model_path = model_path+"d_model_"+str(i)+"/"
	nd_model_path = model_path+"nd_model_"+str(i)+"/"
	
	if not os.path.exists(nd_model_path):
		os.mkdir(nd_model_path)
	if not os.path.exists(d_model_path):
		os.mkdir(d_model_path)

	for j in range(1, n_iter+1):
		d_model_iter_folder = d_model_path+"iter_"+str(j)+"/"
		nd_model_iter_folder = nd_model_path+"iter_"+str(j)+"/"

		if not os.path.exists(d_model_iter_folder):
			os.mkdir(d_model_iter_folder)
		if not os.path.exists(nd_model_iter_folder):
			os.mkdir(nd_model_iter_folder)

		# Train 10 models per depth modification

		# Distilled model
		model = load_model(UNet_modular(channel_depth=2*i, n_channels=3, n_classes=1), device)
		model_descr = "d_model_{}_iteration_{}_param_{}".format(i, j, get_trainable_param(model))
		nd_model, best_dice = model_training(model, train_dataloader, validation_dataloader, train_distilled, evaluate, n_epochs, other_model=teacher_model)
		save_model(model, d_model_iter_folder+model_descr+".pth")
		log_file.write("Dice : {:.3f}\tModel : {}\n".format(best_dice, model_descr))
		log_file.flush()


		# Not Distilled model
		model = load_model(UNet_modular(channel_depth=2*i, n_channels=3, n_classes=1), device)
		model_descr = "nd_model_{}_iteration_{}_param_{}".format(i, j, get_trainable_param(model))
		d_model, best_dice = model_training(model, train_dataloader, validation_dataloader, train, evaluate, n_epochs)
		save_model(model, nd_model_iter_folder+model_descr+".pth")
		log_file.write("Dice : {:.3f}\tModel : {}\n".format(best_dice, model_descr))
		log_file.flush()


log_file.close()