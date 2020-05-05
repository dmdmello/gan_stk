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
    def __init__(self, n_paths_G = 2, latent_dim = 100, img_shape = (1, 28, 28)):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape
        self.n_paths_G = n_paths_G
        self.latent_dim = latent_dim
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        modules = nn.ModuleList()
        for _ in range(self.n_paths_G):
            modules.append(nn.Sequential(
            *block(latent_dim, 128),
            *block(128, 512),
            #*block(256, 512),
            #*block(512, 512),
            #*block(512, 1024),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
            ))
        self.paths = modules

    def forward(self, z):
        img = []
        
        for path in self.paths:
            img.append(path(z).view(img.size(0), *self.img_shape))
        img = torch.cat(img, dim=0)
        return img

class Discriminator(nn.Module):
    def __init__(self, n_paths_G = 2, img_shape = (1, 28, 28)):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(int(np.prod(img_shape)), 512)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)
        modules = nn.ModuleList()
        modules.append(nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
                ))
        modules.append(nn.Sequential(
            nn.Linear(256, n_paths_G),
                ))
        self.paths = modules

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        img_flat = self.lr2(self.fc2(self.lr1(self.fc1(img_flat))))
        validity = self.paths[0](img_flat)
        classifier = F.log_softmax(self.paths[1](img_flat), dim=1)
        return validity, classifier
    
class VAE_MLP(nn.Module):
    def __init__(self, n_paths_G = 2, img_shape = (1, 28, 28), latent_dim = 2):
        super(VAE_MLP, self).__init__()
        
        #encoder
        self.enc_fc1 = nn.Linear(int(np.prod(img_shape)), 512)
        
        #latent
        self.z_mean = nn.Linear(512, latent_dim)
        self.z_log_var = nn.Linear(512, latent_dim)
        
        #decoder
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, int(np.prod(img_shape)))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    def encode(self, x):
        
        enc_fc1 = self.relu(self.enc_fc1(x))
        
        z_mean = self.z_mean(enc_fc1)
        z_log_var = self.z_log_var(enc_fc1)
        
        return z_mean, z_log_var

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        
        return eps.mul(std).add_(mu)

        
    def decode(self, z):
        dec_fc1 = self.relu(self.dec_fc1(z))
        output = self.sigmoid(self.dec_fc2(dec_fc1))
        
        return output

    def forward(self, x):
        x = (x+1)/2
        img_flat = x.view(-1, 784)
        
        mu, logvar = self.encode(img_flat)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Classifier_after_vae(nn.Module):
    def __init__(self, vae, latent_dim = 2, n_paths_G = 2, layer_dim = 32):
        super(Classifier_after_vae, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, n_paths_G)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.vae = vae
        
    def forward(self, img):
        
        recon_batch, mu, logvar = self.vae(img)
        latent = mu
        
        latent_flat = latent.view(latent.size(0), -1)
        logits = self.fc2(self.leaky_relu(self.fc1(latent_flat)))
        
        classifier = F.log_softmax(logits, dim=1)
        
        return classifier
    
class Classifier_independent(nn.Module):
    def __init__(self, n_paths_G = 2, img_shape = (1,28,28), layers_dims = (256,64)):
        super(Classifier_independent, self).__init__()
        
        self.fc1 = nn.Linear(int(np.prod(img_shape)), layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], n_paths_G)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        
    def forward(self, img):
        
        img_flat = img.view(img.size(0), -1)
        fc1 = self.leaky_relu(self.fc1(img_flat))
        fc2 = self.leaky_relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        
        classifier = F.log_softmax(logits, dim=1)
        
        return classifier
    
    
'''class Features_shared(nn.Module):
    def __init__(self, n_paths_G = 2, img_shape = (1, 28, 28)):
        super(Features_shared, self).__init__()
        self.fc1 = nn.Linear(int(np.prod(img_shape)), 512)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)

    def method(self, img): 
        img_flat = img.view(img.size(0), -1)
        features = self.lr2(self.fc2(self.lr1(self.fc1(img_flat))))

        return features
            
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        features = self.lr2(self.fc2(self.lr1(self.fc1(img_flat))))

        return features
    
class Discriminator_shared(nn.Module):
    def __init__(self, feature_dim = 256):
        super(Discriminator_shared, self).__init__()
        
        self.fc1 = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        validity = self.sigmoid(self.fc1(features))

        return validity
    
class Classifier_shared(nn.Module):
    def __init__(self, n_paths_G = 2, feature_dim = 256):
        super(Classifier_shared, self).__init__()
        
        self.fc1 = nn.Linear(feature_dim, n_paths_G)
    
    def forward(self, features):
        
        classification = F.log_softmax(self.fc1(features), dim=1)
        
        return classification
    '''