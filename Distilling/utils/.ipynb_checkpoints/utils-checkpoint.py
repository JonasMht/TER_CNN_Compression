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