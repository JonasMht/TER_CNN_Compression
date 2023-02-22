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
from PIL import Image
from torch import Tensor

from utils.utils import preprocessing
from utils.utils import IoU
from utils.utils import postprocessing



def eval_i3(image, gt, model, device, saving_folder: str, configuration_name: str, nb_slice=150, slice_debut=280, slice_to_save=280) :
    stack_seg = []
    stack_gt = []
    print_it = slice_to_save
    directory_i3 = saving_folder
    img_i3 = image
    gt_i3 = gt

    # On met le modèle en mode eval (on gèle tout)
    model.eval()
    print("model set to eval() mode")
    with torch.no_grad() :
        print("Computing IoU score on LW4->I3 "+configuration_name+" ...")

        #Ici, la range sert à ne prendre que les slices au dessus de la slice 300 pour le test
        for i in range(slice_debut, slice_debut+nb_slice) :
        
            nb_slice = i
        
            if(i%50==0) :
                print("Slice number", i,"...")
        
            image_to_seg_i3, mask_i3 = preprocessing(img_i3[nb_slice], gt_i3[nb_slice], device = device)
    
            segmentation_i3 = postprocessing(F.softmax(model(image_to_seg_i3), dim=1)[0,0].detach().cpu().numpy())
        
            stack_seg.append(segmentation_i3)
            stack_gt.append(mask_i3[0,0,:,:].detach().cpu().numpy())
        
            if (i==print_it) :
            
                plt.axis("off")
                plt.imsave(directory_i3+"/img_to_seg_i3.png", img_i3[nb_slice],cmap="gray")
                #plt.show()
                plt.axis("off")
                plt.imsave(directory_i3+"/pred_i3_"+configuration_name+".png", segmentation_i3)
                #plt.show()
                plt.axis("off")
                plt.imsave(directory_i3+"/gt_i3.png", mask_i3[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                #plt.show()

            del image_to_seg_i3
            del mask_i3
            torch.cuda.empty_cache()
        
    print("IoU Score LW4->I3 "+configuration_name+" : ", IoU(stack_seg, stack_gt))
    print("shape des prédictions", np.shape(stack_seg))
    print("shape des vérités terrains", np.shape(stack_gt))
    print("")


def eval_lw4(image, gt, model, device, saving_folder: str, configuration_name: str):

    stack_seg = []
    stack_gt = []
    showing = False
    directory_lw4 = saving_folder
    img_lw4 = image
    gt_lw4 = gt

    print("Computing IoU score on I3->LW4 "+configuration_name+"...")
    model.eval()
    print("Model set to eval mode")
    with torch.no_grad() :
    
        for j in range(3):
            for i in range(20) :
        
                if(i%20==0) :
                    print("Slice number ", 320+80*j+2*i,"...")
        
                image_to_seg_lw4, mask_lw4 = preprocessing(img_lw4[320+80*j+2*i], gt_lw4[320+80*j+2*i], device = device)
    
                segmentation_lw4 = postprocessing(F.softmax(model(image_to_seg_lw4), dim=1)[0,0].detach().cpu().numpy())
        
                stack_seg.append(segmentation_lw4)
                stack_gt.append(mask_lw4[0,0,:,:].detach().cpu().numpy())
        
                if (i==4 and j==0) :
            
                    plt.axis("off")
                    plt.imsave(directory_lw4+"/img_to_seg_lw4.png", img_lw4[320+80*j+2*i],cmap="gray")
                    #plt.show()
                    plt.axis("off")
                    plt.imsave(directory_lw4+"/pred_lw4_"+configuration_name+".png", segmentation_lw4)
                    #plt.show()
                    plt.axis("off")
                    plt.imsave(directory_lw4+"/gt_lw4.png", mask_lw4[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                    #plt.show()
        
                del image_to_seg_lw4
                del mask_lw4
                torch.cuda.empty_cache()
        
    print("IoU Score I3->LW4 "+configuration_name+" : ",IoU(stack_seg, stack_gt))
    print("shape des prédictions", np.shape(stack_seg))
    print("shape des vérités terrains", np.shape(stack_gt))
    print("")