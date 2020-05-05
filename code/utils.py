import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import seaborn as sn
import pandas as pd

def show(img, rows):
    npimg = img.detach().numpy()
    plt.figure(figsize = (20, rows))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def distribution_select(dist, shape):
    if dist=='uniform':
        return np.random.uniform(-1, 1, shape)
    elif dist=='normal':
        return np.random.normal(0, 1, shape)
    else: 
        return None
    

    
class arg_parser_handler():
    def __init__(self,
        n_epochs = 500,
        batch_size = 100,
        lr = 0.0002,
        b1 = 0.5,
        b2 = 0.999,
        n_cpu = 12,
        latent_dim = 100,
        img_size = 28,
        channels = 1,
        sample_interval = 400,
        n_paths_G = 2,
        classifier_para = 1.0,
        vae_name = None,
        dim1 = 256,
        dim2 = 64,
        min_size_dataset = 10000
                ):
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_cpu = n_cpu
        self.latent_dim = latent_dim 
        self.img_size = img_size
        self.channels = channels
        self.sample_interval = sample_interval 
        self.n_paths_G = n_paths_G
        self.classifier_para = classifier_para
        self.vae_name = vae_name
        self.dim1 = dim1
        self.dim2 = dim2
        self.min_size_dataset = min_size_dataset
        
    def parser_maker(self):
        
        try: 
            parser = argparse.ArgumentParser()
            parser.add_argument('--n_epochs', type=int, default=self.n_epochs, help='number of epochs of training')
            parser.add_argument('--batch_size', type=int, default=self.batch_size, help='size of the batches')
            parser.add_argument('--lr', type=float, default=self.lr, help='adam: learning rate')
            parser.add_argument('--b1', type=float, default=self.b1, help='adam: decay of first order momentum of gradient')
            parser.add_argument('--b2', type=float, default=self.b2, help='adam: decay of first order momentum of gradient')
            parser.add_argument('--n_cpu', type=int, default=self.n_cpu, help='number of cpu threads to use during batch generation')
            parser.add_argument('--latent_dim', type=int, default=self.latent_dim, help='dimensionality of the latent space')
            parser.add_argument('--img_size', type=int, default=self.img_size, help='size of each image dimension')
            parser.add_argument('--channels', type=int, default=self.channels, help='number of image channels')
            parser.add_argument('--sample_interval', type=int, default=self.sample_interval, help='interval betwen image samples')
            parser.add_argument('--n_paths_G', type=int, default=self.n_paths_G, help='number of paths of generator')
            parser.add_argument('--classifier_para', type=float, default=self.classifier_para, help='regularization parameter for classifier')
            
            parser.add_argument('--vae_name', type=str, default=self.vae_name, help='name of pretrained vae')
            parser.add_argument('--dim1', type=int, default=self.dim1, help='classifier first layer dimension')
            parser.add_argument('--dim2', type=int, default=self.dim2, help='classifier second layer dimension')
            parser.add_argument('--min_size_dataset', type=int, default=self.min_size_dataset, help='classifier second layer dimension')
            
            return parser.parse_args(), 0
        except:
            return arg_parser_handler(), 1

        
def generators_confusion_matrix(generator, discriminator, num_gens, shape_z, samples = 100):

    generator.eval()
    discriminator.eval()
    acc_all_gens = np.zeros((num_gens, num_gens))
    temp = []

    for s in range(samples):
        z = Variable(Tensor(distribution_select(DISTRIBUTION, shape_z)))
        for k in range(num_gens):

            # Generate a batch of images
            gen_imgs = generator.paths[k](z)

            # Loss measures generator's ability to fool the discriminator

            validity, classification = discriminator(gen_imgs)
            # Loss measures classifier's ability }to classify various generators

            target = Variable(Tensor(shape_z[0]).fill_(k), requires_grad=False)
            target = target.type(torch.cuda.LongTensor)

            acc_gen_k = []

            for target in range(num_gens):
                acc = ((classification.argmax(dim=1))==target).sum().cpu().numpy()/shape_z[0]
                acc_gen_k.append(acc)

            acc_all_gens[k] = (np.array(acc_all_gens[k]) + np.array(acc_gen_k))
            torch.cuda.empty_cache()

    return acc_all_gens/samples

def plot_confusion_matrix(matrix):
    df_cm = pd.DataFrame(matrix)
    plt.figure(figsize = (10,8))
    sn.heatmap(df_cm, annot=True)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.xlabel('RESPOSTA DO CLASSIFICADOR')
    plt.ylabel('IMAGENS DE CADA GERADOR')
    plt.show() # ta-da!
