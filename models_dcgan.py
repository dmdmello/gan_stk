import argparse
import os
import numpy as np
import math

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm_notebook

import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil
import time



class Generator(nn.Module):
    def __init__(self, n_paths_G = 2, latent_dim = 100, nz=100, nc = 3, ngf=64):
        super(Generator, self).__init__()
        
        self.n_paths_G = n_paths_G
        self.latent_dim = latent_dim

        modules = nn.ModuleList()
        for _ in range(self.n_paths_G):
            modules.append(nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        ))
            
        self.paths = modules

    def forward(self, z):
        img = []
        
        for path in self.paths:
            img.append(path(z))
        img = torch.cat(img, dim=0)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_paths_G = 2):
        super(Discriminator, self).__init__()
        self.n_paths_G = n_paths_G
        self.shared = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
        )
        modules = nn.ModuleList()
        modules.append(
            nn.Sequential(
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, kernel_size=2, stride=2, padding=0, bias=False),
                nn.Sigmoid()
            ))
        modules.append(
            nn.Sequential(
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, self.n_paths_G, kernel_size=2, stride=2, padding=0, bias=False),
                nn.LogSoftmax(dim=1)
            ))
        self.paths = modules

    def forward(self, img):
            
        shared_output = self.shared(img)
        validity = (self.paths[0](shared_output)).view(-1, 1)
        classifier = (self.paths[1](shared_output)).view(-1, self.n_paths_G).squeeze(1)
        
        return validity, classifier
        