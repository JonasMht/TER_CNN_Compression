from __future__ import print_function
#%matplotlib inline
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




def preprocessing(img,mask,device,crop=False) :
    if not isinstance(img, np.ndarray) :
        img = np.asarray(img)
        
    if crop :
        img = img[0:1536-256 , 0:1536-256]
        
    #img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    img = img/255
    img = torch.as_tensor(img.copy()).float().contiguous()
    tmp = torch.randn(1,3,img.shape[1], img.shape[2])
    tmp[0] = img
    img = tmp.to(device=device, dtype=torch.float32)
    
    if not isinstance(mask, np.ndarray) :
        mask = np.asarray(mask)
        
    if crop :
        mask = mask[0:1536-256, 0:1536-256]
        
    mask = np.expand_dims(mask, axis=0)
    #mask = mask/255
    mask_pipe = torch.as_tensor(mask.copy()).float().contiguous()
    mask_pipe_ret = mask_pipe[None,:,:,:].to(device=device, dtype=torch.long)
    
    return img,mask_pipe_ret

def IoU(res, mask) :
    inter = np.logical_and(res, mask)
    union = np.logical_or(res, mask)
    
    iou_score = np.sum(inter) / np.sum(union)
    
    return iou_score

def postprocessing(res_seg) :
    
    res_seg[res_seg < 0.5] = 0
    res_seg[res_seg > 0.5] = 1

    where_0 = np.where(res_seg == 0)
    where_1 = np.where(res_seg == 1) 
 
    res_seg[where_0] = 1
    res_seg[where_1] = 0
    
    return(res_seg)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

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
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def smooth(scalars, weight) :  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed