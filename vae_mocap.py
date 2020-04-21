#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:35:58 2020

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
from lib import mocap_data
import itertools
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
    latent_dim = 16

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 16)
        self._enc_log_sigma = torch.nn.Linear(100, 16)

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

groups = [
  ['root'],
  ['lowerback'],
  ['upperback'],
  ['thorax'],
  ['lowerneck'],
  ['upperneck'],
  ['head'],
  ['rclavicle'],
  ['rhumerus'],
  ['rradius'],
  ['rwrist'],
  ['rhand'],
  ['rfingers'],
  ['rthumb'],
  ['lclavicle'],
  ['lhumerus'],
  ['lradius'],
  ['lwrist'],
  ['lhand'],
  ['lfingers'],
  ['lthumb'],
  ['rfemur'],
  ['rtibia'],
  ['rfoot'],
  ['rtoes'],
  ['lfemur'],
  ['ltibia'],
  ['lfoot'],
  ['ltoes']
]
group_names = [g[0] for g in groups]

joint_order = [joint for grp in groups for joint in grp]

train_trials = [
  (7, 1),
  (7, 2),
  (7, 3),
  (7, 4),
  (7, 5),
  (7, 6),
  (7, 7),
  (7, 8),
  (7, 9),
  (7, 10),
]
test_trials = [
  (7, 11),
  (7, 12)
]

train_trials_data = [
  mocap_data.load_mocap_trial(subject, trial, joint_order=joint_order)
  for subject, trial in train_trials
]
test_trials_data = [
  mocap_data.load_mocap_trial(subject, trial, joint_order=joint_order)
  for subject, trial in test_trials
]
_, joint_dims, _ = train_trials_data[0]
joint_dims['root'] = joint_dims['root'] - 3

Xtrain_raw = torch.FloatTensor(
  # Chain all of the different lists together across the trials
  list(itertools.chain(*[arr for _, _, arr in train_trials_data]))
)[:, 3:]


Xtest_raw = torch.FloatTensor(
  # Chain all of the different lists together across the trials
  list(itertools.chain(*[arr for _, _, arr in test_trials_data]))
)[:, 3:]

# Normalize each of the channels to be within [0, 1].
mins, _ = torch.min(Xtrain_raw, dim=0)
maxs, _ = torch.max(Xtrain_raw, dim=0)
    

mins_test, _ = torch.min(Xtest_raw, dim=0)
maxs_test, _ = torch.max(Xtest_raw, dim=0)



# Some of these things aren't used, and we don't want to divide by zero
Xtrain = (Xtrain_raw - mins) / torch.clamp(maxs - mins, min=0.1)
Xtest = (Xtest_raw - mins_test) / torch.clamp(maxs_test - mins_test, min=0.1)

Xtrain = Xtrain[0:3776,]
input_dim = 59
batch_size = 32
if __name__ == '__main__':
    

    #transform = transforms.Compose(
    #    [transforms.ToTensor()])
    #mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    #dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)

    #print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(16, 100, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    
    for epoch in range(100000):
        Xvar = Variable(Xtrain[torch.randperm(len(Xtrain))[:batch_size]])
        inputs = Xvar
    
    
        optimizer.zero_grad()
        dec = vae(inputs)
        ll = latent_loss(vae.z_mean, vae.z_sigma)
        loss = criterion(dec, inputs) + ll
        loss.backward()
        optimizer.step()
        l = loss.data[0]
        print(epoch, l)
        
# test

test_input = Variable(Xtest)
    
sample = vae(test_input)


def mse(sample):
    import statistics
    n = sample.size(0)
    m = sample.size(1)
    MSE = []
    for i in range(0,n):
        summation = 0
        for j in range(0,m):
    
            diff = Xtest[i][j] - sample[i][j].data.numpy()[-1]
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
    