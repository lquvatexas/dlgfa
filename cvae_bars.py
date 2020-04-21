#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:22:44 2020

@author: luke
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from lib.bars_data import (sample_bars_image, sample_many_bars_images,
                           sample_one_bar_image)
#%matplotlib inline
class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std) + mu
        else:
            return mu

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar

def to_var(x):
    x = Variable(x)
    #if use_cuda:
    #    x = x.cuda()
    return x

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, int(label)] = 1
    return to_var(targets)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD






latent_size = 8 # z dim
image_size = 8

dim_x = image_size * image_size

num_samples = 2000
num_train_samples = 1500
num_test_samples = 500
num_epochs = 20000
batch_size = 64
seq_len = 10

# One bar per image

label = []
data = []
for k in range(num_train_samples):
    i = torch.randperm(image_size)[0]
    A = torch.zeros(image_size, image_size)
    A[i, :] = 0.5
    A += 0.05 * torch.randn(A.size())
    data.append(A.view(-1))
    #i = i.tolist()
    label.append(i)

X = torch.stack([data[i] for i in range(len(data))]) # convert list to tensor
labels = torch.Tensor(label)

test_label = []
test_data = []
for k in range(num_test_samples):
    i = torch.randperm(image_size)[0]
    A = torch.zeros(image_size, image_size)
    A[i, :] = 0.5
    A += 0.05 * torch.randn(A.size())
    test_data.append(A.view(-1))
    #i = i.tolist()
    test_label.append(i)
    
X_test = torch.stack([test_data[i] for i in range(len(test_data))]) # convert list to tensor
test_labels = torch.Tensor(test_label) 

def train(iteration):
    model.train()
    train_loss = 0
    rand = torch.randperm(num_train_samples)[:batch_size]
    Xvar = Variable(X[rand])
    label = labels[rand]
   
    label = one_hot(label, 8)
    recon_batch, mu, logvar = model(Xvar, label)
    optimizer.zero_grad()
    loss = loss_function(recon_batch, Xvar, mu, logvar)
    loss.backward()
    train_loss += loss.data[0]
    optimizer.step()
    if iteration % 500 == 0:
        print('Train iteration: {} \tLoss: {:.6f}'.format(
            iteration, loss.data[0] / len(Xvar)))
def test(epoch):
    model.eval()
    test_loss = 0
    rand = torch.randperm(num_test_samples)[:batch_size]
    Xvar_test = Variable(X_test[rand])
    label_test = test_labels[rand]
 
    label_test = one_hot(label_test, 8)
    recon_batch, mu, logvar = model(Xvar_test, label_test)
    test_loss += loss_function(recon_batch, Xvar_test, mu, logvar).data[0]
    if i == 0:
        n = min(Xvar_test.size(0), 8)
        comparison = torch.cat([Xvar_test[:n],
                                  recon_batch.view(batch_size, 1, 8, 8)[:n]])
        save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(Xvar_test)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# train model
model = CVAE(8*8, latent_size, 8)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for iteration in range(1, 10000):
    train(iteration)
    #test(iteration)

# reconstructed
Xvar_test = Variable(X_test) 
label_test = one_hot(test_labels, 8)
recon_batch, mu, logvar = model(Xvar_test,label_test)

samples = recon_batch
# plot reconstructed images
plt.imshow(samples[2].data.cpu().numpy().reshape(8,8))

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
        




# Generate images with condition labels
c = torch.eye(8, 8) # [one hot labels for 0-7]
c = to_var(c)
z = to_var(torch.randn(8, latent_size))
samples = model.decode(z, c).data.cpu().numpy()

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(10, 10)
gs.update(wspace=0.05, hspace=0.05)
for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(8, 8))





