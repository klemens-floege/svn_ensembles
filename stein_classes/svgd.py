import torch
import torch.autograd as autograd

class SVGD:
    def __init__(self, P, K, optimizer):
        self.P = P
        self.K = K
        self.optim = optimizer
        

    def phi(self, W, X, T):
        W = W.detach().requires_grad_(True)
        
        log_prob = self.P.log_prob(W, X, T)

        score_func = autograd.grad(log_prob.sum(), W)[0]
        
        #print('Score func: ', score_func)
        

        #TODO: maybe put W here 
        K_XX = self.K(W, W.detach())
        grad_K = -autograd.grad(K_XX.sum(), W)[0]

        #print('grad K ', grad_K)
        
        phi = (K_XX.detach().matmul(score_func) + grad_K) / W.size(0)        
        #phi = score_func
        

        return phi, score_func, grad_K, log_prob

    def step(self, W, X, T ):
        self.optim.zero_grad()
        phi_output, score_func, grad_K, log_prob = self.phi(W, X, T)
        if isinstance(phi_output, tuple):
            grad_tensor = phi_output[0]  # Assuming the first element is the desired tensor
        else:
            grad_tensor = phi_output
        
        #W.grad = -grad_tensor
        #print('grad tensor: ', grad_tensor)
        W.grad = -grad_tensor
        #print('W after: ', W)
        
        self.optim.step()
        return phi_output, score_func, grad_K, log_prob