#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:16:26 2020

@author: luke
"""
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from lib.bars_data import sample_one_bar_image

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

image_size = 8 
input_dim = image_size * image_size
batch_size = 32

num_samples = 2000

X = torch.stack([
  sample_one_bar_image(image_size).view(-1)
  for _ in range(num_samples)
])
X += 0.05 * torch.randn(X.size())

num_train_samples = 1500
num_test_samples = 500

if __name__ == '__main__':
    

    #transform = transforms.Compose(
    #    [transforms.ToTensor()])
    #mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    #dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)

    #print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(8, 100, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    
    for epoch in range(100000):
        Xvar = Variable(X[torch.randperm(num_train_samples)[:batch_size]])
        inputs = Xvar
    
    
        optimizer.zero_grad()
        dec = vae(inputs)
        ll = latent_loss(vae.z_mean, vae.z_sigma)
        loss = criterion(dec, inputs) + ll
        loss.backward()
        optimizer.step()
        l = loss.data[0]
        print(epoch, l)
    
    plt.imshow(X[0].numpy().reshape(8, 8))
    plt.imshow(vae(inputs).data[0].numpy().reshape(8, 8))
    plt.show(block=True)

# test
#encoder = Encoder(input_dim, 100, 100)
#decoder = Decoder(8, 100, input_dim)
#vae = VAE(encoder, decoder)
#
#criterion = nn.MSELoss()
#
#optimizer = optim.Adam(vae.parameters(), lr=0.0001)
#l = None 
   
#for epoch in range(50000):
#    X_test = X[num_train_samples:]
#    test_input = Variable(X_test[torch.randperm(num_test_samples)[:batch_size]])
#    
#    optimizer.zero_grad()
#    dec = vae(test_input)
#    ll = latent_loss(vae.z_mean, vae.z_sigma)
#    loss = criterion(dec, inputs) + ll
#    loss.backward()
#    optimizer.step()
#    l = loss.data[0]
#    print(epoch, l)


    
 # reconstructed
X_test = X[num_train_samples:]
test_input = Variable(X_test)
    
sample = vae(test_input)

plt.imshow(vae(test_input).data[0].numpy().reshape(8, 8))
plt.show(block=True)

def mse(sample):
    import statistics
    n = sample.size(0)
    m = sample.size(1)
    MSE = []
    for i in range(0,n):
        summation = 0
        for j in range(0,m):
    
            diff = X_test[i][j] - sample[i][j].data.numpy()[-1]
            squared_diff = diff**2
            summation = summation + squared_diff
        MSE.append(summation/m)
    
    Mean_mse = statistics.mean(MSE)
    STD_mse =  statistics.stdev(MSE)
    
    #loglike = info["loglik_term"].data.numpy()[-1]
            
    print("MSE, STD are", Mean_mse, STD_mse)

def density(sample):
    import seaborn as sns
    import math
    #methods = ["oi-VAE","DLGFA"]
    n = sample.size(0)
    m = sample.size(1)
    mean = []
    var = []
    for i in range(0,n):
        for j in range(0,m):
            mean.append(info["all_dec_mean"][t][i][j].data.numpy()[-1])
            var.append(math.log10(info["all_dec_std"][t][i][j].data.numpy()[-1]))
    #for item in methods:
    sns.distplot(mean,hist=False, kde=True,
                 kde_kws = {'linewidth':2},
                 label = "DLGFA")
    
    
    
    plt.legend(prop={"size":16}, title = "Methods")
    plt.title("Density plot of generated mean")
    #plt.xlabel(")
    plt.ylabel("Density")
        
        

#for i in range(2):
#    save_img_and_reconstruction2(i)
    
    

