import torch
import torch.autograd as autograd
import torch.utils.data as data_utils
import scipy

from laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace

from stein_classes.stein_utils import calc_loss


def apply_SVN(modellist, parameters, 
              batch, train_dataloader, kernel, device, cfg):
    
    inputs = batch[0]
    targets = batch[1]

    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
    

    loss, log_prob = calc_loss(modellist, batch, train_dataloader, cfg)

    score_func = autograd.grad(log_prob.sum(), parameters)
    score_tensors = [t.view(-1) for t in score_func]  # Flatten
    score_func_tensor = torch.cat(score_tensors).view(n_particles, -1)  # (n_particles, n_parameters)


    hessian_particle_loader = data_utils.DataLoader(
        data_utils.TensorDataset(inputs, targets), 
        batch_size=cfg.SVN.hessian_calc_batch_size
        )
    
    
    hessians_list = []

    for i in range(n_particles):
        
        if cfg.SVN.hessian_calc == "Full":  
            laplace_particle_model = FullLaplace(modellist[i], likelihood='regression')
            
        elif cfg.SVN.hessian_calc == "Diag":  
            laplace_particle_model = DiagLaplace(modellist[i], likelihood='regression')

        elif cfg.SVN.hessian_calc == "Kron":  
            laplace_particle_model = KronLaplace(modellist[i], likelihood='regression')

        elif cfg.SVN.hessian_calc == "LowRank":  
            laplace_particle_model = LowRankLaplace(modellist[i], likelihood='regression')
            
        else: 
            ValueError("This is type of Hessian calculation is not yet implemented")
        
        laplace_particle_model.fit(hessian_particle_loader)
        Hessian = laplace_particle_model.posterior_precision
        hessians_list.append(Hessian)

    hessians_tensor = torch.cat(hessians_list, dim=0)
    hessians_tensor = hessians_tensor.reshape(n_particles, n_parameters, n_parameters) #(n_particles, n_parameters, n_parameters)

    #compute curvature matrix

    if cfg.SVN.use_curvature_kernel == "use_curvature":
        M = torch.mean(hessians_tensor, axis=0)
    else:
        M=None

    param_tensors = [p.view(-1) for p in parameters]  # Flatten
    ensemble_parameters_tensor = torch.cat(param_tensors).view(n_particles, -1)  # Concatenate and reshape


    h = kernel.sigma
    
    displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
    
    if M is not None:
        U = torch.linalg.cholesky(M)
        ensemble_parameters_tensor = torch.matmul(U, ensemble_parameters_tensor.T).T
        displacement_tensor = torch.matmul(displacement_tensor, M)
    squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
    K_XX = torch.exp(-squared_distances / h)
    grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h

    
    
    v_svgd = -1 * torch.einsum('mn, mo -> no', K_XX, score_func_tensor) / n_particles + torch.mean(grad_K, dim=0)
    
    
    H1 = torch.einsum("xy, xz, xbd -> yzbd", K_XX, K_XX, hessians_tensor) #(n_particles, n_particles, n_parametes_per_model, n_parametes_per_model)
    H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_parametes_per_model, n_parametes_per_model))
    H1[range(n_particles), range(n_particles)] += H2

    N, _, D, _ = H1.shape  # Extract N and D from the shape of H
    H_reshaped = H1.permute(0, 2, 1, 3).reshape(N*D, N*D)  # Reshape to [N*D, N*D] ,  #(n_particles*n_parametes_per_model, n_particles*n_parametes_per_model)
    H =  H_reshaped / n_particles

    #implemented as in SVN paper
    #v_svgd = t2
    #H = hessians_tensor# / n_particles

    solve_method = 'CG'
    # solve_method = 'Cholesky'

    if solve_method == 'CG':
        
        cg_maxiter = 50
        H_numpy = H.detach().cpu().numpy()

        v_svgd_numpy = v_svgd.detach().cpu().flatten().numpy()

        alphas, _ = scipy.sparse.linalg.cg(H_numpy, v_svgd_numpy, maxiter=cg_maxiter)
        alphas = torch.tensor(alphas, dtype=torch.float32).reshape(n_particles, n_parameters).to(device)
                
        alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, n_parameters)
        v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)


    elif solve_method == 'Cholesky':
        lamb = 0.01
        H = H1 + NK * lamb
        UH = scipy.linalg.cholesky(H)
        v_svn = self._getSVN_direction(kx, v_svgd, UH)
    
    else: 
        ValueError("This equations system solver is not implemented")
    

    #Assign gradients    
    for model, grads in zip(modellist, v_svn):
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

    return loss
    