import torch
import numpy as np

class Ensemble(torch.nn.Module):
    def __init__(self, device, net, particles=None, n_particles=1):
        super(Ensemble, self).__init__()
        self.net = net
        if particles is None:
            self.particles = (1*torch.randn(n_particles, *torch.Size([self.net.num_params]))).to(device)
            print("initialised particles: ", self.particles.shape)
        else:
            self.particles = particles

        self.weighs_split = [np.prod(w) for w in net.param_shapes]

    def reshape_particles(self, z):
        reshaped_weights = []
        z_splitted = torch.split(z, self.weighs_split, 1)
        for j in range(z.shape[0]):
            l = []
            for i, shape in enumerate(self.net.param_shapes):
                l.append(z_splitted[i][j].reshape(shape))
            reshaped_weights.append(l)
        return reshaped_weights

    def forward(self, x, W=None):
        if W is None:
            W = self.particles
        models = self.reshape_particles(W)
        if self.net.out_act is None:
            pred = [self.net.forward(x, w) for w in models]
            return [torch.stack(pred)] #.unsqueeze(0)
        else:
            pred,hidden = zip(*(list(self.net.forward(x,w)) for w in models))
            return torch.stack(pred), torch.stack(hidden)