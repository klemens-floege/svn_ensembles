import torch
import torch.autograd as autograd

from stein_classes.loss import calc_loss

def apply_SVGD(modellist, parameters, 
              batch, train_dataloader, kernel, device, cfg):
    
    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
    

    loss, log_prob = calc_loss(modellist, batch, train_dataloader, cfg, device)

    #print("loss: ", loss.shape)
    #print("loss sum", loss.sum())

        

    score_func = autograd.grad(log_prob.sum(), parameters)
    score_tensors = [t.view(-1) for t in score_func]  # Flatten
    score_func_tensor = torch.cat(score_tensors).view(n_particles, -1)  # (n_particles, n_parameters)    

    #Gradient Clipping
    if cfg.experiment.grad_clip: 
        norm = score_func_tensor.norm(p=2, dim=1, keepdim=True)  # Compute the norm for each particle
        max_norm = cfg.experiment.grad_clip_max_norm
        norm_clipping_mask = (norm > max_norm)
        norm_clipping_mask = norm_clipping_mask.squeeze()

        for i in range(norm_clipping_mask.shape[0]):
            if bool(norm_clipping_mask[i]):
                score_func_tensor[i] = (score_func_tensor[i] 
                                                / norm[i]) * max_norm
    


    param_tensors = [p.view(-1) for p in parameters]  # Flatten
    ensemble_parameters_tensor = torch.cat(param_tensors).view(n_particles, -1)  # Concatenate and reshape

        
    displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
    
    h = kernel.sigma
    squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
    K_XX = torch.exp(-squared_distances / h)

    

    grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h
    

    
    #phi = (K_XX.detach().matmul(score_func_tensor) + grad_K) / n_particles   

    v_svgd = -1 * torch.einsum('mn, mo -> no', K_XX, score_func_tensor) / n_particles + torch.mean(grad_K, dim=0) 

    #print(v_svgd)   

    #print('svgd v ', v_svgd)

    

    #Assign gradients    
    for model, grads in zip(modellist, v_svgd):
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

     # Gradient clipping
    #max_norm = 2.0  # Set a maximum norm for the gradients; adjust as needed
    #torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    gradients = [p.grad for p in parameters][0]
    #print('gradients: ', gradients)

    return loss