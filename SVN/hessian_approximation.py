from abc import ABC, abstractmethod
import torch


class HessianApproximation(ABC):
    def __init__(self, model, hessian_particle_loader, cfg, device, optimizer, step):
        self.modellist = model
        self.cfg = cfg
        self.hessian_particle_loader = hessian_particle_loader
        self.device = device
        self.optimizer = optimizer
        self.step = step


        self.n_particles = len(self.modellist)
        self.n_parameters = sum(p.numel() for p in self.modellist[0].parameters() if p.requires_grad)

        self.cg_maxiter = 50

        #self.hessian_list = None


    @abstractmethod
    def compute_hessians(self):
        pass

    @abstractmethod
    def compute_kernel(self):
        pass

    @abstractmethod
    def solve_linear_system(self, K_XX, grad_K, v_svgd):
        pass

    @abstractmethod
    def solve_block_linear_system(self, K_XX, grad_K, v_svgd):
        pass

    def assign_grad(self, v_svn):
        for model, grads in zip(self.modellist, v_svn):
            model.to(self.device)
            # Flatten all parameters of the model and their gradients
            flat_params = torch.cat([p.view(-1) for p in model.parameters()])
            
            # Ensure we're iterating over the actual parameters of the model
            param_iter = model.parameters()
            
            grad_idx = 0  # Start index for slicing gradients from grads
            for param in param_iter:
                # Number of elements in the current parameter
                num_param_elements = param.numel()
                
                # Slice the gradient for the current parameter
                grad_for_param = grads[grad_idx:grad_idx + num_param_elements].view(param.size())
                
                # Manually assign the gradient for the current parameter
                if param.grad is not None:
                    param.grad = grad_for_param
                else:
                    param.grad = grad_for_param.clone()
                
                # Update the start index for the next parameter
                grad_idx += num_param_elements
        
