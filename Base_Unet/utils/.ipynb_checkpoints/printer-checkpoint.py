import torch
import tifffile
import torch.utils.data as data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def source_printer(dataloader_source, device) :
    real_batch = next(iter(dataloader_source))


    print("images source : ", real_batch["image"].shape)
    print("mask source :", real_batch["mask"].shape)
                       
    fig = plt.figure(figsize=(10,10)) # specifying the overall grid size
    fig.suptitle("training exemple source", fontsize=20)

    plt.subplot(2,1,1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(real_batch["image"][0,0].cpu(), cmap="gray")
    plt.subplot(2,1,2)
    plt.imshow(real_batch["mask"][0,0].cpu(), cmap="gray")
    #print(real_batch["mask"][0])
    #plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    real_batch = next(iter(dataloader_source))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training images source - a batch")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    
def target_printer(dataloader_target, device) :
    real_batch = next(iter(dataloader_target))
    print("images target : ", real_batch["image"].shape)
    print("mask target :", real_batch["mask"].shape)
                       
    fig = plt.figure(figsize=(10,10)) # specifying the overall grid size
    fig.suptitle("training exemple target", fontsize=20)

    plt.subplot(2,1,1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(real_batch["image"][0,0].cpu(), cmap="gray")
    plt.subplot(2,1,2)
    plt.imshow(real_batch["mask"][0,0].cpu(), cmap="gray")
    #print(real_batch["mask"][0])
    #plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    real_batch = next(iter(dataloader_target))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training images target - a batch")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))