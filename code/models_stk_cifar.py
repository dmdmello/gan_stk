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

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Generator(nn.Module):
    def __init__(self, n_paths_G = 2, latent_dim = 100, nz=100, nc = 3, ngf=128):
        super(Generator, self).__init__()
        
        self.n_paths_G = n_paths_G
        self.latent_dim = latent_dim
        
        modules = nn.ModuleList()
        for _ in range(self.n_paths_G):
            modules.append(nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(latent_dim, ngf * 2 * 8 * 8),
            nn.ReLU(True),
            Reshape(-1, ngf * 2, 8, 8),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
            # output size. (3) x 32 x 32
        ))
            
        self.paths = modules

    def forward(self, z):
        img = []
        
        for path in self.paths:
            img.append(path(z))
        img = torch.cat(img, dim=0)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=128, n_paths_G = 2, init_sample = None):
        super(Discriminator, self).__init__()
        
        self.n_paths_G = n_paths_G
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.shared = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf, momentum = 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2, momentum = 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
        )
        modules = nn.ModuleList()
        modules.append(
            nn.Sequential(                
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4, momentum = 0.9),
                nn.LeakyReLU(0.2, inplace=True)
                
                # state size. (ndf*4) x 4 x 4 
            ))
        modules.append(
            nn.Sequential(                
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4, momentum = 0.9),
                nn.LeakyReLU(0.2, inplace=True),
               
                # state size. (ndf*4) x 4 x 4
            ))
        
        self.paths = modules
        
        self.shared
        '''for i, ly in enumerate(disc.shared):
            #print(type(ly) == torch.nn.modules.conv.Conv2d)
            if (type(ly) == torch.nn.modules.conv.Conv2d):
                shape = self.shared[:i+1](init_sample).shape
                print('layer: {}'.format(i))
                print(ly)
                print('output shape: {}'.format(shape))'''
            
        total_units = self.paths[0](self.shared(init_sample)).view(init_sample.shape[0], -1).shape[-1]
        print("Total Units : {}".format(total_units))
        
        self.linear_disc  = nn.Linear(total_units, 1)
        self.linear_class = nn.Linear(total_units, n_paths_G)
        
    def forward(self, img):
            
        shared_output = self.shared(img)
        validity = (self.paths[0](shared_output)).view(img.shape[0], -1)
        validity = self.sigmoid(self.linear_disc(validity))
        
        classification = (self.paths[1](shared_output)).view(img.shape[0], -1)
        classification = self.linear_class(classification)
        classification = self.log_softmax(classification)
        
        #validity = (self.paths[0](shared_output)).view(-1, 1)
        #classifier = (self.paths[1](shared_output)).view(-1, self.n_paths_G).squeeze(1)
        
        return validity, classification
    
    