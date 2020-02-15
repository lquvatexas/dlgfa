
""" dlgfa setup on the metabolomics data.

* Point estimates on the sparse weight parameters.
* Proximal gradient steps are taken to handle the group lasso penalty.
"""


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



torch.manual_seed(0)


dim_z = 30
group_input_dim = 40
dim_h = 100
prior_theta_scale = 1
lam = 5
lam_adjustment = 1

num_epochs = 20000
mc_samples = 1
batch_size = 5
n_layers = 1

groups = [
  ['control'],
  ['cold'],
  ['heat'],
  ['lactose'],
  ['oxidative'],
  ]


group_names = [g[0] for g in groups]


# data load
# This function will allow load data/model from an older version of pytorch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

metrain_raw = torch.load("data/metabo/mtrain_raw.pt")

# Normalize each of the channels to be within [0, 1].
mins, _ = torch.min(metrain_raw, dim=0)
maxs, _ = torch.max(metrain_raw, dim=0)

# Some of these things aren't used, and we don't want to divide by zero
metrain = (metrain_raw - mins) / torch.clamp(maxs - mins, min=0.1)

dataloader = torch.utils.data.DataLoader(
  # TensorDataset is stupid. We have to provide two tensors.
  torch.utils.data.TensorDataset(metrain_raw, torch.zeros(metrain_raw.size(0))),
  batch_size=batch_size,
  shuffle=True
)

dim_x = metrain.size(2)
num_groups = len(groups)
#group_dims = 196

stddev_multiple = 1

group_dims = [196 for grp in groups]
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
    
    
seq_len = metrain_raw.size(0)

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

vae = OIVAE(
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

plot_interval = 500
elbo_per_iter = []
iteration = 0
for epoch in range(num_epochs):
  
  X = metrain[:,torch.randperm(8)[:batch_size]]
  #for Xbatch, _ in dataloader:
  if iteration > 1000:
    stddev_multiple = 2
    #X = metrain_raw[:,torch.randperm(8)[:batch_size]]
  info = vae.step(
      X=Variable(X),
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

