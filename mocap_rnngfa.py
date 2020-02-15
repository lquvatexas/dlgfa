#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:48:01 2020

@author: luke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:48:01 2020

@author: luke
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

# Necessary for the torch.utils.data stuff.
import torchvision

from lib import mocap_data
from lib.distributions import Normal
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.oivae import NormalPriorTheta, OIVAE
from lib.utils import Lambda


torch.manual_seed(0)

dim_z = 16
group_input_dim = 8
dim_h = 100
prior_theta_scale = 1
lam = 5
lam_adjustment = 1

num_epochs = 70
mc_samples = 1
batch_size = 32
n_layers = 1


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
dataloader = torch.utils.data.DataLoader(
  # TensorDataset is stupid. We have to provide two tensors.
  torch.utils.data.TensorDataset(Xtrain, torch.zeros(Xtrain.size(0))),
  batch_size=batch_size,
  shuffle=True
)

dim_x = Xtrain.size(1)
num_groups = len(groups)
group_dims = [sum(joint_dims[j] for j in grp) for grp in groups]

stddev_multiple = 0.1

# inference_net_log_stddev = Variable(
#   torch.log(1e-2 * torch.ones(dim_z)),
#   requires_grad=True
# )
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


seq_len = 32

generative_net = BayesianGroupLassoGenerator(
  seq_len = seq_len,
  group_generators=[make_group_generator(dim) for dim in group_dims],
  group_input_dim=group_input_dim,
  dim_z=dim_z,
  dim_h = dim_h
)

def debug_z_by_group_matrix(t):
    
    
    # groups x dim_z
  fig, ax = plt.subplots(dpi=200)
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
  
  
  
lr = 1e-3
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
#  {'params': [inference_net_log_stddev], 'lr': lr},
  {'params': generative_net.group_generators_parameters(), 'lr': lr},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr}
])





Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
])

vae = OIVAE(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws],
  dim_x = dim_x,
  dim_h = dim_h,
  dim_z = dim_z,
  n_layers = n_layers
)

plot_interval = 500
elbo_per_iter = []
iteration = 0
for epoch in range(num_epochs):
  for Xbatch, _ in dataloader:
    if iteration > 1000:
      stddev_multiple = 1
      print()
    Xbatch=torch.stack([Xbatch for _ in range(20)])
    Xbatch = Xbatch.transpose(0,1)
    info = vae.step(
      X=Variable(Xbatch),
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
    
def save_amc(filename, seq):
  ordered_joint_dims = [joint_dims[joint] for joint in joint_order]
  joint_ixs_start = np.cumsum([0] + ordered_joint_dims)

  with open(filename, 'w') as f:
    # preamble
    f.write('#!OML:ASF\n')
    f.write(':FULLY-SPECIFIED\n')
    f.write(':DEGREES\n')

    for i in range(seq.size(0)):
      # Keyframe number, starting at 1
      f.write('{}\n'.format(i + 1))

      for joint, dim, ix_start in zip(joint_order, ordered_joint_dims, joint_ixs_start):
        f.write('{} '.format(joint))

        # The root has three special channels for the XYZ translation that we do
        # not learn.
        if joint == 'root':
          f.write('0 0 0 ')

        f.write(' '.join([str(seq[i, ix_start + j]) for j in range(dim)]))
        f.write('\n')
#X = Xbatch
#Xvar = Variable(X)
#Xvar = Variable(X[:,torch.randperm(num_train_samples)[:batch_size]])
#info = vae.step(
#    X=Xvar,
#    prox_step_size=Ws_lr * lam * lam_adjustment,
#    mc_samples=mc_samples
#  )


#reconstructed = generative_net(inference_net(Variable(X)).mu).mu.data * torch.clamp(maxs - mins, min=0.1) + mins

#save_amc('07_reconstructed.amc', reconstructed)
    
X = Xtrain[0:32,:]
#Xbatch_test = Xtest[0:10,0:1,] 
Xv = torch.stack([X for _ in range(20)]).transpose(0,1)
info_test = vae.step(
      X=Variable(Xv),
      prox_step_size=Ws_lr * lam * lam_adjustment,
     mc_samples=mc_samples
)
#def save_img_and_reconstruction_mo(i):

#mins_m = torch.stack([mins for _ in range(20)])
#maxs_m = torch.stack([maxs for _ in range(20)])
reconstructed = (info_test["all_gr2"][1]).data * torch.clamp(maxs - mins, min=0.1) + mins
save_amc('07_1.amcf3', reconstructed)

info_test["loglik_term"] 
 # for test data walking 
 
Xt1 = Xtest[0:32,:]
Xt_test1 = torch.stack([Xt1 for _ in range(20)]).transpose(0,1)
info_test1 = vae.step(
      X=Variable(Xt_test1),
      prox_step_size=Ws_lr * lam * lam_adjustment,
     mc_samples=mc_samples
)
reconstructed1 = (info_test1["all_gr2"][30]).data * torch.clamp(maxs_test - mins_test, min=0.1) + mins_test
save_amc('07_30testw.amc', reconstructed1)
# likelihood
info_test1["loglik_term"]

# test data side walking 

Xt2 = Xtest[316:348,:]
Xt_test2 = torch.stack([Xt2 for _ in range(20)]).transpose(0,1)
info_test2 = vae.step(
      X=Variable(Xt_test2),
      prox_step_size=Ws_lr * lam * lam_adjustment,
     mc_samples=mc_samples
)
reconstructed2 = (info_test2["all_gr2"][1]).data * torch.clamp(maxs_test - mins_test, min=0.1) + mins_test
save_amc('07_1tests.amc', reconstructed2)

#likelihood
info_test2["loglik_term"]


