import numpy as np
from scipy import  optimize
import torch

import torch.nn as nn



class TorchGaussian(nn.Module):
    def __init__(self):
        super(TorchGaussian, self).__init__()
        self.DoF = 2
        self.nData = 1
        self.mu0 = torch.zeros(self.DoF, 1)
        self.std0 = torch.ones(self.DoF, 1)
        self.var0 = self.std0 ** 2
        self.stdn = 0.3
        self.varn = self.stdn ** 2
        torch.manual_seed(40)
        self.A = torch.randn(self.DoF, 1)
        self.thetaTrue = torch.randn(self.DoF)
        self.data = self.simulateData()


    def forward(self, thetas):
       return self.getMinusLogPosterior(thetas)
   
    def getForwardModel(self, thetas):
       thetas = torch.tensor(thetas)
       nSamples = thetas.numel() // self.DoF
       thetas = thetas.view(self.DoF, nSamples)
       tmp = torch.sum(self.A * thetas, 0)
       return tmp if nSamples > 1 else tmp.squeeze()
   
    def getJacobianForwardModel(self, thetas):
       thetas = torch.tensor(thetas)
       nSamples = thetas.numel() // self.DoF
       thetas = thetas.view(self.DoF, nSamples)
       tmp = self.A.repeat(1, nSamples)
       return tmp if nSamples > 1 else tmp.squeeze()
   
    def simulateData(self):
       noise = torch.randn(1, self.nData) * self.stdn
       return self.getForwardModel(self.thetaTrue) + noise
   
    def getMinusLogPrior(self, thetas):
       nSamples = thetas.numel() // self.DoF
       thetas = thetas.view(self.DoF, nSamples)
       shift = thetas - self.mu0
       tmp = 0.5 * torch.sum(shift ** 2 / self.var0, 0)
       return tmp if nSamples > 1 else tmp.squeeze()
   
    def getMinusLogLikelihood(self, thetas):
       nSamples = thetas.numel() // self.DoF
       thetas = thetas.view(self.DoF, nSamples)
       F = self.getForwardModel(thetas)
       shift = F - self.data
       tmp = 0.5 * torch.sum(shift ** 2, 0) / self.varn
       return tmp if nSamples > 1 else tmp.squeeze()
   
    def getMinusLogPosterior(self, thetas):
       return self.getMinusLogPrior(thetas) + self.getMinusLogLikelihood(thetas)
   

    def getGradientMinusLogPrior(self, thetas):
        thetas = torch.tensor(thetas)

        nSamples = thetas.numel() // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        tmp = (thetas - self.mu0) / self.var0
        return tmp if nSamples > 1 else tmp.squeeze()
   

    def getGradientMinusLogLikelihood(self, thetas, *arg):
        thetas = torch.tensor(thetas)
        nSamples = thetas.numel() // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        F = arg[0] if len(arg) > 0 else self.getForwardModel(thetas)   
        J = arg[1] if len(arg) > 1 else self.getJacobianForwardModel(thetas)
        tmp = J * np.sum(F - self.data, 0) / self.varn
        return tmp if nSamples > 1 else tmp.squeeze()
   

    def getGradientMinusLogPosterior(self, thetas, F, J):
       return self.getGradientMinusLogPrior(thetas) + self.getGradientMinusLogLikelihood(thetas)

    def getGNHessianMinusLogPosterior(self, thetas):
       nSamples = thetas.numel() // self.DoF
       thetas = thetas.view(self.DoF, nSamples)
       J = self.getJacobianForwardModel(thetas)
       tmp = J.view(self.DoF, 1, nSamples) * J.view(1, self.DoF, nSamples) / self.varn \
             + torch.eye(self.DoF).unsqueeze(2) / self.var0
       return tmp if nSamples > 1 else tmp.squeeze()
   
    def getMAP(self, initial_guess=None):
        if initial_guess is None:
            x0 = torch.randn(self.DoF)
        else:
            x0 = torch.tensor(initial_guess, dtype=torch.float32)
        x0_np = x0.numpy()
        res = optimize.minimize(lambda x: self.getMinusLogPosterior(torch.from_numpy(x)).item(), x0_np, method='L-BFGS-B')
        return torch.from_numpy(res.x)
