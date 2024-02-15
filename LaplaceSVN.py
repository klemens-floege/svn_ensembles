import numpy as np
import torch

#from laplace import hessian

class SVN:
    def __init__(self, model, *arg):
        self.model = model
        self.DoF = model.DoF
        self.nParticles = 100
        self.nIterations = 100
        self.stepsize = 1
        self.MAP = torch.unsqueeze(self.model.getMAP(torch.randn(self.DoF, dtype=torch.float32)), dim=1)
        if len(arg) == 0:
            self.resetParticles()
        else:
            self.particles = arg[0]

    def apply(self):
        maxmaxshiftold = np.inf
        maxshift = np.zeros(self.nParticles)
        Q = torch.zeros((self.DoF, self.nParticles))

        print("start applz")

        for iter_ in range(self.nIterations):

            
            F = self.model.getForwardModel(self.particles)
            J = self.model.getJacobianForwardModel(self.particles)


            print(F.shape)
            print(J.shape)

            gmlpt = self.model.getGradientMinusLogPosterior(self.particles, F, J)

            # Use Laplace library to compute Hessian
            #Hmlpt = hessian(lambda x: -self.model.getLogPosterior(x, F, J), self.particles)
            Hmlpt  = self.model.getGNHessianMinusLogPosterior(self.particles, J)


            M = np.mean(Hmlpt, 2)

            for i_ in range(self.nParticles):
                sign_diff = self.particles[:, i_, np.newaxis] - self.particles
                Msd = np.matmul(M, sign_diff)
                kern = np.exp(-0.5 * np.sum(sign_diff * Msd, 0))
                gkern = Msd * kern

                mgJ = np.mean(-gmlpt * kern + gkern, 1)
                HJ = np.mean(Hmlpt * kern ** 2, 2) + np.matmul(gkern, gkern.T) / self.nParticles
                Q[:, i_] = np.linalg.solve(HJ, mgJ)
                maxshift[i_] = np.linalg.norm(Q[:, i_], np.inf)
            self.particles += self.stepsize * Q
            maxmaxshift = np.max(maxshift)

            if np.isnan(maxmaxshift) or (maxmaxshift > 1e20):
                print('Reset particles...')
                self.resetParticles()
                self.stepsize = 1
            elif maxmaxshift < maxmaxshiftold:
                self.stepsize *= 1.01
            else:
                self.stepsize *= 0.9
            maxmaxshiftold = maxmaxshift

    def resetParticles(self):
        self.particles = np.random.normal(scale=1, size=(self.DoF, self.nParticles))
