import torch
import math

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y, M=None):

        if M is None:
            M = torch.eye(X.size(1), device=X.device, dtype=X.dtype)

        XX = (X.matmul(M) * X).sum(dim=1, keepdim=True)
        YY = (Y.matmul(M) * Y).sum(dim=1, keepdim=True).t()
        XY = X.matmul(M.matmul(Y.t()))

        dnorm2 = XX + YY - 2 * XY

        # Apply the median heuristic
        if self.sigma is None:
            sigma = self.median(dnorm2.detach()) / (2 * torch.tensor(math.log(X.size(0) + 1), device=X.device))
        else:
            sigma = self.sigma ** 2

        gamma = 1.0 / (2 * sigma)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

