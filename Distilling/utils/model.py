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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, norm_layer, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if norm_layer == "bn":

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        elif norm_layer == "in":

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer, bilinear=True, skip_connection=True):
        super().__init__()

        self.skip_connection = skip_connection

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, norm_layer)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_layer)

    def forward(self, x1, x2):

        if self.skip_connection:
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

        else:
            x1 = self.up(x1)
            return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, norm_layer="bn", bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm_layer = norm_layer

        self.inc = DoubleConv(n_channels, 64, norm_layer)
        self.down1 = Down(64, 128, norm_layer)
        self.down2 = Down(128, 256, norm_layer)
        self.down3 = Down(256, 512, norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_layer)
        self.up1 = Up(1024, 512 // factor, norm_layer, bilinear)
        self.up2 = Up(512, 256 // factor, norm_layer, bilinear)
        self.up3 = Up(256, 128 // factor, norm_layer, bilinear)
        self.up4 = Up(128, 64, norm_layer, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return logits
        # return [d1,d2,d3,d4,u1,u2,u3,u4], logits


# To do
class UNet_small(nn.Module):
    def __init__(self, n_channels, n_classes, norm_layer="bn", bilinear=False):
        super(UNet_small, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm_layer = norm_layer

        self.inc = DoubleConv(n_channels, 64, norm_layer)
        self.down1 = Down(64, 128, norm_layer)
        self.down2 = Down(128, 256, norm_layer)
        self.down3 = Down(256, 512, norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_layer)
        self.up1 = Up(1024, 512 // factor, norm_layer, bilinear)
        self.up2 = Up(512, 256 // factor, norm_layer, bilinear)
        self.up3 = Up(256, 128 // factor, norm_layer, bilinear)
        self.up4 = Up(128, 64, norm_layer, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        # d3 = self.down3(d2)
        # d4 = self.down4(d3)
        # u1 = self.up1(d4, d3)
        # u2 = self.up2(u1, d2)
        u3 = self.up3(d2, d1)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return logits
        # return [d1,d2,d3,d4,u1,u2,u3,u4], logits


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Recons_net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Recons_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_classes, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class YNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(YNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear, skip_connection=False)
        self.up2 = Up(512, 256 // factor, bilinear, skip_connection=False)
        self.up3 = Up(256, 128 // factor, bilinear, skip_connection=False)
        self.up4 = Up(128, 64, bilinear, skip_connection=False)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.up1(x, x)
        x2 = self.up2(x1, x1)
        x3 = self.up3(x2, x2)
        x4 = self.up4(x3, x3)
        return self.outc(x4)


class ClassifNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ClassifNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output    # return x for visualization


class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden,
                      training=self.is_training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_hidden,
                      training=self.is_training)
        x = self.fc3(x)
        return x


class StudentNetwork(nn.Module):
    def __init__(self):
        super(StudentNetwork, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden,
                      training=self.is_training)
        x = self.fc2(x)
        return x


class StudentNetworkSmall(nn.Module):
    def __init__(self):
        super(StudentNetworkSmall, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 30)
        self.fc2 = nn.Linear(30, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden,
                      training=self.is_training)
        x = self.fc2(x)
        return x


class GenericNetwork(nn.Module):
    def __init__(self, first_layer_size, second_layer_size):
        super(GenericNetwork, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, first_layer_size)
        self.fc2 = nn.Linear(30, second_layer_size)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden,
                      training=self.is_training)
        x = self.fc2(x)
        return x
