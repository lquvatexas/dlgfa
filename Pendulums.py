#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:08:28 2020

@author: luke
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

# Necessary for the torch.utils.data stuff.
import torchvision

from lib.distributions import Normal
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.dlgfa import NormalPriorTheta, dlgfa
from lib.utils import Lambda

# Initial and end valus
st = 0
et = 20.4
ts = 0.1
g = 9.81
L = 1
b = 0.5
m = 1

def sim_pen_eq(t,theta):
    dtheta2_dt = (-b/m)*theta[1] + (-g/L)*np.sin(theta[0])
    dtheta1_dt = theta[1]
    return [dtheta1_dt,dtheta2_dt]


# data generation


n = 200
theta1_ini = 0
theta2_ini = 3
theta3_ini = 5
theta4_ini = 7
theta5_ini = 9
theta6_ini = 10
theta_ini = [theta5_ini,theta6_ini]
t_span = [st,et+ts]
t = np.arange(st,et+ts,ts)
sim_points = len(t)
l = np.arange(0,sim_points,1)

theta14 = solve_ivp(sim_pen_eq, t_span, theta_ini,t_eval = t)
theta5 = theta14.y[0,:]
theta6 = theta14.y[1,:]

#
#plt.plot(t, theta5, label="Angular Displacement (rad)")
#plt.plot(t, theta6, label="Angular velocity (rad/s)")
#
#
#
plt.xlabel("Time(s)")
plt.ylabel("Angular Disp.(rad) and Angular Vel.(rad/s)")
plt.legend()
plt.show()

theta1 = theta12.y[0,:]
theta2 = theta12.y[1,:]

theta3 = theta13.y[0,:]
theta4 = theta13.y[1,:]

theta5 = theta14.y[0,:]
theta6 = theta14.y[1,:]

t1 = np.repeat(theta1,200)
t2 = np.repeat(theta2,200)
t3 = np.repeat(theta3,200)
t4 = np.repeat(theta4,200)
t5 = np.repeat(theta5,200)
t6 = np.repeat(theta6,200)


lorenz = []
for i in range(50):
        temp = np.c_[torch.Tensor(t1[i*200:200*(i+1)]),torch.Tensor(t2[i*200:200*(i+1)]),torch.Tensor(t3[i*200:200*(i+1)]),torch.Tensor(t4[i*200:200*(i+1)]),torch.Tensor(t5[i*200:200*(i+1)]),torch.Tensor(t6[i*200:200*(i+1)])]
        lorenz.append(torch.Tensor(temp))
 
lozdat = Variable(torch.stack(lorenz,0)) 

lozdat = lozdat.transpose(1,2)
lozdatm = lozdat.contiguous().view(200, -1)
lozdatm2 = lozdatm.contiguous().view(20,10,300)

dim_z = 10
group_input_dim = 50
dim_h = 100
prior_theta_scale = 1
lam = 50
lam_adjustment = 1

num_epochs = 30000
mc_samples = 1
#batch_size = 2
n_layers = 1


groups = [
  ['g1'],
  ['g2'],
  ['g3'],
  ['g4'],
  ['g5'],
  ['g6']
  ]


group_names = [g[0] for g in groups]
dim_x = lozdatm2.size(2)
num_groups = len(groups)
#group_dims = 196

stddev_multiple = 1

group_dims = [50 for grp in groups]

inference_net = NormalNet(
  mu_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_h + dim_h, dim_z)
  ),
sigma_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_h + dim_h, dim_z),
    Lambda(torch.exp),
    Lambda(lambda x: x * stddev_multiple + 1e-3)
  )
)

def make_group_generator(output_dim):
  # Note that this Variable is NOT going to show up in `net.parameters()` and
  # therefore it is implicitly free from the ridge penalty/p(theta) prior.
  log_sigma = Variable(
    torch.log(1e-2 * torch.ones(output_dim)),
    requires_grad=True
  )
  return NormalNet(
    mu_net=torch.nn.Sequential(
      # torch.nn.Linear(group_input_dim, 16),
      torch.nn.Tanh(),
      torch.nn.Linear(group_input_dim, output_dim)
    ),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )
    
seq_len = lozdatm2.size(0)

generative_net = BayesianGroupLassoGenerator(
  seq_len = seq_len,
  group_generators=[make_group_generator(dim) for dim in group_dims],
  group_input_dim=group_input_dim,
  dim_z=dim_z,
  dim_h = dim_h
)

def debug_z_by_group_matrix(t):
    
    # groups x dim_z
  fig, ax = plt.subplots()
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.Ws[t].data, 2), dim=2)
  )
  W_col_norms_prop = W_col_norms / torch.max(W_col_norms, dim=0)[0]
  ax.imshow(W_col_norms_prop.numpy(), aspect='equal')
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  ax.yaxis.set_ticks(np.arange(len(groups)))
  ax.yaxis.set_ticklabels(group_names)
  
  
  
lr = 1e-4
betas = (0.9,0.999)

lr_inferencenet = 1e-4
betas_inferencenet = (0.9,0.999)

lr_generativenet = 1e-4
betas_generativenet = (0.9,0.999)


optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr_inferencenet, 'betas':betas_inferencenet},
  # {'params': [inference_net_log_stddev], 'lr': lr},
  {'params': generative_net.group_generators_parameters(), 'lr': lr_generativenet,'betas': betas_generativenet},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr, 'betas':betas}
])

    
Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
])

vae = dlgfa(
  inference_model=inference_net,
  generative_model=generative_net,
  #prior_z=prior_z,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws],
  dim_x = dim_x,
  dim_h = dim_h,
  dim_z = dim_z,
  n_layers = n_layers
)

randtrain = torch.randperm(10)[:8]
m_train = lozdatm2[:,randtrain]
m_test = lozdatm2[:,list(set([i for i in range(10)]) - set(randtrain))]


plot_interval = 500
elbo_per_iter = []
iteration = 0
batch_size = 5
for epoch in range(num_epochs):
  
  X = m_train[:,torch.randperm(8)[:batch_size]]
  #for Xbatch, _ in dataloader:
  if iteration > 1000:
    stddev_multiple = 2
    #X = metrain_raw[:,torch.randperm(8)[:batch_size]]
  info = vae.step(
      X=X,
      prox_step_size=Ws_lr * lam * lam_adjustment,
      mc_samples=mc_samples
    )
  elbo_per_iter.append(info['elbo'].data[0])
    
  if iteration % plot_interval == 0 and iteration > 0:
    debug_z_by_group_matrix(8)

    plt.figure()
    plt.plot(elbo_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO per iteration. lam = {}'.format(lam))
    plt.show()  
      
  print('epoch', epoch, 'iteration', iteration)
  print('  ELBO:', info['elbo'].data[0])
  print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
  print('    loglik_term      ', info['loglik_term'].data[0])
  print('    log p(theta)     ', info['logprob_theta'].data[0])
  print('    log p(W)         ', info['logprob_W'].data[0])

  iteration += 1

print('Outputting reconstructions AMC file...') 


test_input = m_test
info_test = vae.step(
    X=test_input,
    prox_step_size=Ws_lr * lam * lam_adjustment,
    mc_samples=mc_samples
  )
 
import statistics
      
def mse(t):
    
    n = 8
    m = 300
    MSE = []
    for i in range(0,n):
        summation = 0
        for j in range(0,m):
            diff = m_train[t][i][j].data.numpy()[-1] - info["all_gr2"][t][i][j].data.numpy()[-1]
            squared_diff = diff**2
            summation = summation + squared_diff
        MSE.append(summation/m)
    
    Mean_mse = statistics.mean(MSE)
    STD_mse =  statistics.stdev(MSE)
    
    loglike = info_test["loglik_term"].data.numpy()[-1]
            
    print("MSE, STD, loglike are", Mean_mse, STD_mse, loglike)



def mse_test(t):
    
    n = 2
    m = 300
    MSE = []
    for i in range(0,n):
        summation = 0
        for j in range(0,m):
    
            diff = m_test[t][i][j].data.numpy()[-1] - info_test["all_gr2"][t][i][j].data.numpy()[-1]
            squared_diff = diff**2
            summation = summation + squared_diff
        MSE.append(summation/m)
    
    Mean_mse = statistics.mean(MSE)
    STD_mse =  statistics.stdev(MSE)
    
    loglike = info_test["loglik_term"].data.numpy()[-1]
            
    print("MSE, STD  and loglike are", Mean_mse, STD_mse, loglike)



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
        


  






     
#lozdat= lozdat.transpose(0,2,1)
