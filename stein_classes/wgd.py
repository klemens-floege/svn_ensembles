import torch
import torch.autograd as autograd
import torch.utils.data as data_utils
import scipy
import numpy as np

from laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace

from stein_classes.loss import calc_loss
from stein_classes.stein_utils import hessian_matvec, kfac_hessian_matvec_block, diag_hessian_matvec_block


def apply_WGD(modellist, parameters, batch, train_dataloader, 
              kernel, device, cfg, epoch, ann_sch):
    # WGD based on the paper
    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
    

    loss,log_prob = calc_loss(modellist, batch, train_dataloader, cfg, device)
    # print('######## loss:',loss)
        

    score_func = autograd.grad(log_prob.sum(), parameters)
    score_tensors = [t.view(-1) for t in score_func]  # Flatten
    score_func_tensor = torch.cat(score_tensors).view(n_particles, -1)  # (n_particles, n_parameters)

    param_tensors = [p.view(-1) for p in parameters]  # Flatten
    ensemble_parameters_tensor = torch.cat(param_tensors).view(n_particles, -1)  # Concatenate and reshape

        
    displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
    
    h = kernel.sigma
    squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
    K_XX = torch.exp(-squared_distances / h)

    # gradient of the repulsive term i.e. K_XX
    grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h

    
    wgd = -score_func_tensor + (torch.sum(grad_K,dim=0)/(torch.sum(K_XX,dim=0).reshape(-1,1)))*ann_sch[epoch-1]


    

    #Assign gradients    
    for model, grads in zip(modellist, wgd):
        # Flatten all parameters of the model and their gradients
        flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        
        # Ensure we're iterating over the actual parameters of the model
        param_iter = model.parameters()
        
        grad_idx = 0  # Start index for slicing gradients from grads
        for param in param_iter:
            # Number of elements in the current parameter
            num_param_elements = param.numel()
            
            # Slice the gradient for the current parameter
            grad_for_param = grads[grad_idx:(grad_idx + num_param_elements)].view(param.size())
            
            # Manually assign the gradient for the current parameter
            if param.grad is not None:
                param.grad = grad_for_param
            else:
                param.grad = grad_for_param.clone()
            
            # Update the start index for the next parameter
            grad_idx += num_param_elements

    return loss