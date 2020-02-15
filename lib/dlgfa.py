import torch
from torch.autograd import Variable
import torch.nn as nn

from .distributions import Normal
from .utils import KL_Normals


class NormalPriorTheta(object):
  """A distribution that places a zero-mean Normal distribution on all of the
  `group_generators` in a BayesianGroupLassoGenerator."""

  def __init__(self, sigma):
    self.sigma = sigma

  def logprob(self, module):
    return sum(
      Normal(
        torch.zeros_like(param),
        self.sigma * torch.ones_like(param)
      ).logprob(param)
      for gen in module.group_generators
      for param in gen.parameters()
    )

class dlgfa(object):
    def __init__(
      self,
      inference_model,
      generative_model,
      #prior_z,
      prior_theta,
      lam,
      optimizers,
      dim_x,
      dim_h,
      dim_z,
      n_layers,bias=False
  ):
        self.inference_model = inference_model
        self.generative_model = generative_model
        #self.prior_z = prior_z
        self.prior_theta = prior_theta
        self.lam = lam
        self.optimizers = optimizers
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.dim_z = dim_z
        self.n_layers = n_layers
        
        
        self.phi_x = nn.Sequential(
            nn.Linear(dim_x, dim_h))
            

        self.phi_z = nn.Sequential(
            nn.Linear(dim_z, dim_h),
            nn.ReLU())
        #prior for z 
        self.prior = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU())
        self.prior_mean = nn.Linear(dim_h, dim_z)
        self.prior_std = nn.Sequential(
            nn.Linear(dim_h, dim_z),
            nn.Softplus())
        
        
        #recurrence
        self.rnn = nn.GRU(dim_h + dim_h, dim_h, n_layers, bias)
        
        
    def step(self, X, prox_step_size, mc_samples):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_gr2 = []
        
        z_kl = 0
        loglik_term = 0
        logprob_W = 0
        logprob_theta = 0
        h = Variable(torch.zeros(self.n_layers, X.size(1), self.dim_h))
        
        for t in range(X.size(0)):
      
            batch_size = X.size(1)
     
    
            phi_x_t = self.phi_x(X[t])
    
            q_z_t = self.inference_model(torch.cat([phi_x_t,h[-1]],1))
            
            
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            
            zs_t = self.generative_model.Ws[t]
    
           
            z_kl += self._kld_gauss(q_z_t.mu, q_z_t.sigma, prior_mean_t, prior_std_t) / batch_size
    
           
            
            z_sample_t = torch.cat([q_z_t.sample() for _ in range(mc_samples)], dim=0)
            phi_z_t = self.phi_z(z_sample_t)
            
            Xrep = Variable(X[t].data.repeat(mc_samples, 1))
      
            loglik_term += (
                    self.generative_model(phi_z_t,zs_t,h[-1]).logprob(Xrep)
                    / mc_samples
                    / batch_size
                    )

   
            
            logprob_theta = self.prior_theta.logprob(self.generative_model)

    
            logprob_W += -self.lam * self.generative_model.group_lasso_penalty(zs_t)


            loss = -1.0 * (-z_kl + loglik_term + logprob_theta)
            elbo = -loss + logprob_W
            
            enc_mean_t = q_z_t.mu
            enc_std_t = q_z_t.sigma
            
            g_t = self.generative_model(phi_z_t,zs_t,h[-1])
            dec_mean_t = g_t.mu
            dec_std_t = g_t.sigma
            
            gr_t2 = self.generative_model(phi_z_t,zs_t,h[-1]).sample()
            
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            
            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            all_gr2.append(gr_t2)
            
            for opt in self.optimizers:
                opt.zero_grad()
                loss.backward(retain_graph=True)
            for opt in self.optimizers:
                opt.step()
            if self.lam > 0:
                self.generative_model.proximal_step(zs_t,prox_step_size)

        return {
      #'q_z': q_z,
      'z_kl': z_kl,
      #'z_sample': z_sample,
      'loglik_term': loglik_term,
      'logprob_theta': logprob_theta,
      'logprob_W': logprob_W,
      'loss': loss,
      'elbo': elbo,
      'all_enc_mean': all_enc_mean,
      'all_enc_std': all_enc_std,
      'all_dec_mean': all_dec_mean,
      'all_dec_std': all_dec_std,
      'all_gr2':all_gr2
      
      
      }

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		#Using std to compute KLD#

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)
    