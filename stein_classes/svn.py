import numpy as np
import torch
import torch.autograd as autograd
from laplace import FullLaplace, KronLaplace, DiagLaplace
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data_utils
from backpack import backpack, extend
from backpack.extensions import BatchGrad


class SVN:
    def __init__(self, P, K, optimizer):
        self.P = P
        self.K = K
        self.optim = optimizer
    

    def phi(self, W, X, T):
        
        #h = 2*self.DoF , DOF ? 

        W = W.detach().requires_grad_(True)

        model = self.P.ensemble.net

        #model = extend(model)
        #lossfunc = extend(lossfunc)
        
        log_prob = self.P.log_prob(W, X, T)
        
        print('log_prob: ', log_prob)
        

        #with backpack(BatchGrad()):
        #    loss = log_prob.sum()
        #    loss.backward()

        #for name, parameter in model.named_parameters():
        #    print(f"{name:>20}'s grad_batch shape: {parameter.grad_batch.shape}")
        
        score_func = autograd.grad(log_prob.sum(), W)[0]
        #gmlpt  =  score_func 

        #GN_Hmlpt = self.model.getGNHessianMinusLogPosterior_ensemble(X)
        #M = np.mean(GN_Hmlpt, axis=0)        

        #Create a DataLoader for the single particle
        print('length X', X.shape[0])

        print('X: ', X)
        print('T: ', T)

        #particle_loader = DataLoader(TensorDataset(X, T), batch_size=2)
        particle_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X, T), 
            batch_size=2
            )
        
        print('length dataloader', len(particle_loader))
        #model = self.P.ensemble.net
        param_list = [p for p in model.parameters()]
        #print('W' ,W)
        print(param_list)

        for param in model.parameters():
            print(hasattr(param, 'grad_batch'))

        #laplace_particle_model = FullLaplace(self.P.ensemble.net, likelihood='regression')
        laplace_particle_model = DiagLaplace(self.P.ensemble.net, likelihood='regression')

        #for l in param_list:
        #    print('grad: ', l.grad_batch )

        laplace_particle_model.fit(particle_loader)
        

        Hessian = laplace_particle_model.posterior_precision

        print(Hessian)
        print(Hessian.shape)
        
        K_XX = self.K(W, W.detach())
        grad_K = -autograd.grad(K_XX.sum(), W)[0]
        
        phi = (K_XX.detach().matmul(score_func) + grad_K) / W.size(0)        

        return phi, score_func, grad_K, log_prob

    def step(self, W, X, T ):
        self.optim.zero_grad()
        phi_output, score_func, grad_K, log_prob = self.phi(W, X, T)
        if isinstance(phi_output, tuple):
            grad_tensor = phi_output[0]  # Assuming the first element is the desired tensor
        else:
            grad_tensor = phi_output
        
        W.grad = -grad_tensor
        
        self.optim.step()
        return phi_output, score_func, grad_K, log_prob