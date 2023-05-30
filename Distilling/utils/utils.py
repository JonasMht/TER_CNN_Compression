from __future__ import print_function
# %matplotlib inline
import argparse
import os
import os.path as osp
import random
import torch
import torch.nn as nn
import tifffile
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image
from tqdm import tqdm
from torch import Tensor
from datetime import datetime
import copy # To make a copy of a model

def preprocessing(img, mask, device, crop=False):
	if not isinstance(img, np.ndarray):
		img = np.asarray(img)

	if crop:
		img = img[0:1536-256, 0:1536-256]

	# img = np.transpose(img, (2,0,1))
	img = np.expand_dims(img, axis=0)
	img = img/255
	img = torch.as_tensor(img.copy()).float().contiguous()
	tmp = torch.randn(1, 3, img.shape[1], img.shape[2])
	tmp[0] = img
	img = tmp.to(device=device, dtype=torch.float32)

	if not isinstance(mask, np.ndarray):
		mask = np.asarray(mask)

	if crop:
		mask = mask[0:1536-256, 0:1536-256]

	mask = np.expand_dims(mask, axis=0)
	# mask = mask/255
	mask_pipe = torch.as_tensor(mask.copy()).float().contiguous()
	mask_pipe_ret = mask_pipe[None, :, :, :].to(
		device=device, dtype=torch.long)

	return img, mask_pipe_ret


def IoU(res, mask):
	inter = np.logical_and(res, mask)
	union = np.logical_or(res, mask)

	iou_score = np.sum(inter) / np.sum(union)

	return iou_score


def postprocessing(res_seg):

	res_seg[res_seg < 0.5] = 0
	res_seg[res_seg > 0.5] = 1

	where_0 = np.where(res_seg == 0)
	where_1 = np.where(res_seg == 1)

	res_seg[where_0] = 1
	res_seg[where_1] = 0

	return (res_seg)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
	# Average of Dice coefficient for all batches, or for a single mask
	assert input.size() == target.size()
	if input.dim() == 2 and reduce_batch_first:
		raise ValueError(
			f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

	if input.dim() == 2 or reduce_batch_first:
		inter = torch.dot(input.reshape(-1), target.reshape(-1))
		sets_sum = torch.sum(input) + torch.sum(target)
		if sets_sum.item() == 0:
			sets_sum = 2 * inter

		return (2 * inter + epsilon) / (sets_sum + epsilon)
	else:
		# compute and average metric for each batch element
		dice = 0
		for i in range(input.shape[0]):
			dice += dice_coeff(input[i, ...], target[i, ...])
		return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
	# Average of Dice coefficient for all classes
	assert input.size() == target.size()
	dice = 0
	for channel in range(input.shape[1]):
		dice += dice_coeff(input[:, channel, ...], target[:,
						   channel, ...], reduce_batch_first, epsilon)

	return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
	# Dice loss (objective to minimize) between 0 and 1
	assert input.size() == target.size()
	fn = multiclass_dice_coeff if multiclass else dice_coeff
	return 1 - fn(input, target, reduce_batch_first=True)

def smooth(scalars, weight):  # Weight between 0 and 1
	last = scalars[0]  # First value in the plot (first timestep)
	smoothed = list()
	for point in scalars:
		smoothed_val = last * weight + \
			(1 - weight) * point  # Calculate smoothed value
		smoothed.append(smoothed_val)						# Save it
		# Anchor the last smoothed value
		last = smoothed_val

	return smoothed


def kd_loss(output, soft_targets, gt, alpha = 0.5, T = 1):
	# Compute soft targets using the teacher model
	with torch.no_grad():
		soft_targets = F.softmax(soft_targets/T, dim=1)
	
	loss = dice_loss(output, gt)
	distill_loss = dice_loss(F.softmax(output/T, dim=1), soft_targets) * (T * T) * alpha + loss * (1. - alpha)
	return distill_loss

# Alternative approach
def kd_loss2(output, soft_targets, gt, alpha=0.9, T=5):
    """
    Compute the knowledge distillation loss.

    Arguments:
    - output: Output logits from the student network (torch.Tensor)
    - soft_targets: Soft targets (probabilities) from the teacher network (torch.Tensor)
    - gt: Ground truth labels (torch.Tensor)
    - alpha: Weighting factor for the soft target loss (float, default=0.5)
    - T: Temperature parameter for softmax (float, default=1)

    Returns:
    - kd_loss: Knowledge distillation loss (torch.Tensor)
    """
    # Compute the cross-entropy loss between the output and ground truth
    ce_loss = F.cross_entropy(output, gt)

    # Apply softmax with temperature T to the output and soft targets
    output_soft = F.softmax(output / T, dim=1)
    soft_targets_soft = F.softmax(soft_targets / T, dim=1)

    # Compute the Kullback-Leibler (KL) divergence loss between the output and soft targets
    kl_loss = F.kl_div(output_soft.log(), soft_targets_soft, reduction='batchmean')

    # Compute the overall knowledge distillation loss
    kd_loss = (1 - alpha) * ce_loss + alpha * T * T * kl_loss

    return kd_loss

# My utils

def load_model(model, device, path=""):
	ngpu = torch.cuda.device_count()

	model.to(device=device)

	if (device.type == 'cuda'):
		print("Data Parallel")
		model = nn.DataParallel(model, list(range(ngpu)))
	
	if len(path)>0:
		model.load_state_dict(torch.load(path))
	
	return model


# Other

def save_model(model, path):
	torch.save(model.state_dict(), path)

# Function that returns a list of paths in a folder that start with a word and that are sorted in increasing order
def get_path_list(path, start_with=""):
	path_list = []
	for file in os.listdir(path):
		if file.startswith(start_with):
			path_list.append(os.path.join(path, file))
	path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return path_list
# Function that returns in a path list with the first number in the name

def get_trainable_param(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	return sum([np.prod(p.size()) for p in model_parameters])


def show_prediction(model, val_loader, n=0):
	model.eval().cuda()

	with torch.no_grad():
		for i,(img,gt) in enumerate(val_loader):
			if n !=0 and i > n:
				break
		
			if torch.cuda.is_available():
				img, gt = img.cuda(), gt.cuda()
			img, gt = Variable(img), Variable(gt)

			output = model(img)
			output = output.clamp(min = 0, max = 1)
			gt = gt.clamp(min = 0, max = 1)
			

			# Plot image, output and ground truth one one row
			fig, ax = plt.subplots(1,3, figsize=(15,15))
			ax[0].imshow(img[0].cpu().numpy().transpose(1,2,0))
			ax[1].imshow(output[0].cpu().numpy().transpose(1,2,0))
			ax[2].imshow(gt[0].cpu().numpy().transpose(1,2,0))
			# Name each axis
			ax[0].set_title("Image")
			ax[1].set_title("Output")
			ax[2].set_title("Ground Truth")
			
			plt.show()
	

# Training and testing

def evaluate(model, val_loader):
	model.eval().cuda()

	dl = []
	with torch.no_grad():
		for i,(img,gt) in enumerate(val_loader):
			if torch.cuda.is_available():
				img, gt = img.cuda(), gt.cuda()
			img, gt = Variable(img), Variable(gt)

			output = model(img)
			output = output.clamp(min = 0, max = 1)
			gt = gt.clamp(min = 0, max = 1)
			dice_loss = dice_coeff(output, gt)
			dl.append(dice_loss.item())
	
	mean_dice = np.mean(dl)
	return mean_dice


def train(model, optimizer, train_loader, other_model=None):
	model.train().cuda()
	dl = []
	with tqdm(total=len(train_loader)*train_loader.batch_size, desc=f'Training', unit='img') as pbar:
		for i, (inputs, labels) in enumerate(train_loader):
			#print('i', i)
			if torch.cuda.is_available():
				inputs, labels = inputs.cuda(), labels.cuda()
			
			inputs, labels = Variable(inputs), Variable(labels)

			output = model(inputs)
			output = output.clamp(min = 0, max = 1)
			labels = labels.clamp(min = 0, max = 1)
			loss = dice_loss(output, labels)
			dice = dice_coeff(output, labels)
			dl.append(dice.item())

			
			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

			pbar.update(len(inputs))
			pbar.set_postfix(**{'loss': loss.item(), "dice": dice.item()})
	
	mean_dice = np.mean(dl)

	print("Average dice {:.3f} over this epoch.".format(mean_dice))

	return mean_dice


# Training function where the teacher model is used to generate soft targets in the case of image segmentation
def train_distilled(model, optimizer, train_loader, other_model=None):
	T = 5  # temperature for distillation loss
	# Using a higher value for T produces a softer probability distribution over classes
	alpha = 0.9
	# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
	# loss = alpha * st + (1 - alpha) * tt

	teacher_model = other_model

	model.train().cuda()
	dl = []
	with tqdm(total=len(train_loader)*train_loader.batch_size, desc=f'Training', unit='img') as pbar:
		for i, (img, gt) in enumerate(train_loader):
			#print('i', i)
			if torch.cuda.is_available():
				img, gt = img.cuda(), gt.cuda()
			
			img, gt = Variable(img), Variable(gt)

			# Work on this to transfer all code into the kd_loss function++++++++++++-

			student_output = model(img)
			student_output = student_output.clamp(min = 0, max = 1)

			teacher_output = teacher_model(img)
			teacher_output = teacher_output.clamp(min = 0, max = 1)

			gt = gt.clamp(min = 0, max = 1)

			dice = dice_coeff(student_output, gt)
			dl.append(dice.item())


			# Compute the distillation loss
			distillation_loss = kd_loss(student_output, teacher_output, gt, alpha, T)

			optimizer.zero_grad()
			distillation_loss.backward()

			optimizer.step()

			pbar.update(len(img))
			pbar.set_postfix(**{'loss': distillation_loss.item(), "dice": dice.item()})

	mean_dice = np.mean(dl)

	print("Average dice {:.3f} over this epoch.".format(mean_dice))

	return mean_dice

import math

# From https://github.com/udemirezen/continual-learning-benchmark/blob/958fe2645e4c3f8b56b03ce5cc7c6f5d35874a71/train/customized_distill_trainer.py#L202
def get_locality_preserving_alpha(teacher_features, i, j, k=5):
        sigma = math.sqrt(2) # normalizing constant
        f_T_i = teacher_features[i]
        dist = torch.norm(teacher_features - f_T_i, dim=1, p=None)
        knn_indices = dist.topk(k+1, largest=False).indices[1:] # 0th index is always the element itself
        if j in knn_indices:
            alpha_i_j = - dist[j].float().pow(2) / sigma ** 2
            alpha_i_j = torch.exp(alpha_i_j).item()
        else:
            alpha_i_j = 0.
        return alpha_i_j

def locality_preserving_loss(student_features, teacher_features, labels, k=5, lambda_lpl=0.2):
    # locality preserving loss
	k, gamma = 5 if len(labels) > 5 else math.ceil(len(labels) / 2), 1.5
	lpl_loss = 0
	for i, data in enumerate(student_features):
		f_s_i = data
		for j, data_ in enumerate(student_features):
			if j != i:
				alpha_i_j = get_locality_preserving_alpha(teacher_features, i, j, k)
				if alpha_i_j > 0:
					temp_ = torch.norm(f_s_i - data_, dim=0, p=None).pow(2)
					lpl_loss += temp_.item() * alpha_i_j
	
	lpl_loss = gamma * lpl_loss / (labels.shape[0] * k)  # scale by factor: gamma / (k * batch_size)
	loss += lpl_loss
	loss_stats += f" LPL loss: {lpl_loss}"

# Train each distilled model layer on the error of each later of the teacher model
def train_feature_embedding(student_model, optimizer, train_loader, teacher_model=None):
	T = 1  # temperature for distillation loss
	# Using a higher value for T produces a softer probability distribution over classes
	alpha = 0.95
	# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
	# loss = alpha * st + (1 - alpha) * tt

	teacher_model.eval().cuda()
	student_model.train().cuda()
	running_loss = 0

	"""
	features_train = []
	labels_train = []
	for i, (inputs, labels) in enumerate(train_loader):
		if torch.cuda.is_available():
			inputs, labels = inputs.cuda(), labels.cuda()
		features = teacher_model(inputs)
		features_train.append(features.detach())
		labels_train.append(labels)
	
	features_train = torch.cat(features_train, dim=0)
	labels_train = torch.cat(labels_train, dim=0)

	with tqdm(total=len(train_loader), desc=f'Training', unit='img') as pbar:
		for i, (inputs, labels) in enumerate(zip(features_train, labels_train), 0):
			if torch.cuda.is_available():
				inputs, labels = inputs.cuda(), labels.cuda()
			optimizer.zero_grad()
			outputs = student_model(inputs)
			loss = dice_loss(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			
			pbar.update(1)
			pbar.set_postfix(**{'loss': loss.item(), "dice": running_loss/(i+1)})

	
	"""	
	with tqdm(total=len(train_loader), desc=f'Training', unit='img') as pbar:
		for i, (inputs, labels) in enumerate(train_loader):
			
			if torch.cuda.is_available():
				inputs, labels = inputs.cuda(), labels.cuda()

			optimizer.zero_grad()

			feat = student_model(inputs)
			feat_t = teacher_model(inputs)
			# Implement criterion

			#loss, = dice_loss(feat, labels)
			#loss_t = dice_loss(feat_t, labels)
			

			loss = locality_preserving_loss(feat, feat_t, labels, k=5, lambda_lpl=0.2)
			
			

			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			pbar.update(1)
			pbar.set_postfix(**{'loss': loss.item(), "dice": running_loss/(i+1)})

	mean_dice = running_loss/len(train_loader)
	return mean_dice

def model_training(model, train_loader, val_loader, trainFun, evalFun, n_epochs, name="", other_model=None, early_stopping=10):
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

	former_best_dice = 0 # Worst possible
	early_stopping_streak = 0

	scheduled_saves = 0
	best_model = model

	for epoch in range(n_epochs):
		if epoch % 10 == 9:
			scheduled_saves += 1
		print(" --- training: epoch {}".format(epoch+1))
		
		# Train the model
		train_dice = trainFun(model, optimizer, train_loader, other_model)

		# Modulate training rate
		scheduler.step()

		#evaluate for one epoch on validation set
		valid_dice = evalFun(model, val_loader)
		print("Eval metrics : dice {:.3f}.".format(valid_dice))

		

		if former_best_dice < valid_dice:
			# dice is increasing
			early_stopping_streak = 0
			best_model = copy.deepcopy(model)
			if scheduled_saves and name != "":
				scheduled_saves = 0
				torch.save(model.state_dict(), "{}_epoch_{}.pth".format(name, epoch+1))
		else:
			# Is decresing
			early_stopping_streak += 1
			if early_stopping_streak >= early_stopping:
				# 15 consecutive epochs without improvement in dice
				print("Early stopping at epoch {}".format(epoch+1))
				break
		
		former_best_dice = max(former_best_dice, valid_dice)
		
	return (best_model, former_best_dice)
