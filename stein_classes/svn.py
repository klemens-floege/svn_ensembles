import torch
import torch.autograd as autograd
import torch.utils.data as data_utils
import scipy
import numpy as np

from laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace
from laplace.curvature import AsdlGGN, AsdlEF, GGNInterface
from laplace import Laplace

from stein_classes.loss import calc_loss
from stein_classes.stein_utils import hessian_matvec, kfac_hessian_matvec_block, diag_hessian_matvec_block

import inspect


def apply_SVN(modellist, parameters, 
              batch, train_dataloader, kernel, device, cfg, optimizer, step):
    
    inputs = batch[0].to(device)
    targets = batch[1].to(device)

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

    
    #TODO; squee targt maybe
    if cfg.task.dataset in ['mnist', 'fashionmnist'] : 
        inputs_squeezed = inputs #do not squeeue [Bsz, 1, 28, 28]
    else:
        inputs_squeezed = inputs.squeeze(1)

    
    if cfg.SVN.classification_likelihood and cfg.task.task_type == 'classification':
        #print(targets.shape)
        targets = targets.squeeze(1)
        if inputs_squeezed.dtype != torch.float:
            inputs_squeezed = inputs_squeezed.float()
        if targets.dtype != torch.long:
            targets = targets.long()

    #print(targets.shape)
    '''targets = targets.squeeze(1)
    if inputs_squeezed.dtype != torch.float:
        inputs_squeezed = inputs_squeezed.float()
    if targets.dtype != torch.long:
        targets = targets.long()'''

    hessian_particle_loader = data_utils.DataLoader(
        data_utils.TensorDataset(inputs_squeezed, targets), 
        batch_size=cfg.SVN.hessian_calc_batch_size
        )
    
    hessians_list = []
    diag_hessian_list = []
    kron_list = []

    for i in range(n_particles):
        #likelihood_type= cfg.task.task_type

        if cfg.SVN.hessian_calc == "Full":  
            if cfg.SVN.classification_likelihood:
                laplace_particle_model = FullLaplace(modellist[i], 
                                                 likelihood=cfg.task.task_type)
            else:
                laplace_particle_model = FullLaplace(modellist[i], 
                                                 likelihood='regression')
            
                laplace_particle_model.fit(hessian_particle_loader)
                Hessian = laplace_particle_model.posterior_precision
            hessians_list.append(Hessian)
            
        elif cfg.SVN.hessian_calc == "Diag":  

            if cfg.SVN.use_adam_hessian and 1 < step:
                second_moments = []
                for state in optimizer.state.values():
                    if 'exp_avg_sq' in state:
                        second_moments.append(state['exp_avg_sq'])
                
                beta2 = optimizer.param_groups[0]['betas'][1]

                t = step  # Assuming t = 1 for simplicity, update accordingly

                # Calculate v_t
                v_hat_list = [v_t * (1 - beta2 ** t) for v_t in second_moments]

                # Flatten each v_t and concatenate into a single tensor
                flattened_v_hat_list = [v_hat.flatten() for v_hat in v_hat_list]

                result_tensor = torch.cat(flattened_v_hat_list)
                hessians_tensor = result_tensor.view(n_particles, -1)

            
            else:
                if cfg.SVN.classification_likelihood:
                    laplace_particle_model = DiagLaplace(modellist[i], 
                                                    likelihood=cfg.task.task_type)
                else:
                    laplace_particle_model = DiagLaplace(modellist[i], 
                                                    likelihood='regression')
                laplace_particle_model.fit(hessian_particle_loader)
                Hessian = laplace_particle_model.posterior_precision
                diag_hessian_list.append(Hessian)
            
      

        elif cfg.SVN.hessian_calc == "Kron":  
            #backend = AsdlGGN if args.approx_type == 'ggn' else AsdlEF
            #TODO: double check AsdL
            if cfg.SVN.ll:
                laplace_particle_model = Laplace(modellist[i], 'regression')
                        #subset_of_weights='last_layer',
                        #hessian_structure='Full'
                            
                laplace_particle_model.fit(hessian_particle_loader)
            else:
                laplace_particle_model = KronLaplace(modellist[i], 
                                                 likelihood='regression'                                                 
                                                 )
                laplace_particle_model.fit(hessian_particle_loader)
            kron_decomposed = laplace_particle_model.posterior_precision
            kron_list.append(kron_decomposed)
            
        else: 
            ValueError("This is type of Hessian calculation is not yet implemented")
        
    
    if cfg.SVN.hessian_calc == "Full":  
        hessians_tensor = torch.cat(hessians_list, dim=0)
        hessians_tensor = hessians_tensor.reshape(n_particles, n_parameters, n_parameters) #(n_particles, n_parameters, n_parameters)

    if cfg.SVN.hessian_calc == "Diag":  
        if cfg.SVN.use_adam_hessian and 1 < step:
            hessians_tensor = hessians_tensor
        else: 
            hessians_tensor = torch.cat(diag_hessian_list, dim=0)
            hessians_tensor = hessians_tensor.reshape(n_particles, n_parameters) #(n_particles, n_parameters)

    

    #compute curvature matrix
    #TODO: implement kernel M for KFAC, Diag, 
    if cfg.SVN.use_curvature_kernel == "use_curvature":
        M = torch.mean(hessians_tensor, axis=0)
    else:
        M=None

   

    param_tensors = [p.view(-1) for p in parameters]  # Flatten
    ensemble_parameters_tensor = torch.cat(param_tensors).view(n_particles, -1)  # Concatenate and reshape


    h = kernel.sigma
    
    displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
    
    if M is not None:
        if cfg.SVN.hessian_calc == "Full":  
            epsilon = 1e-6  # A small value; adjust as necessary
            M_reg = M + epsilon * torch.eye(M.size(0), device=M.device)
            U = torch.linalg.cholesky(M_reg)
            #U = torch.linalg.cholesky(M)
            ensemble_parameters_tensor = torch.matmul(U, ensemble_parameters_tensor.T).T
            displacement_tensor = torch.matmul(displacement_tensor, M)
        elif cfg.SVN.hessian_calc == "Diag":  
            ensemble_parameters_tensor = M * ensemble_parameters_tensor
            displacement_tensor = displacement_tensor * M
        
    squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
    K_XX = torch.exp(-squared_distances / h) #(n_particles, n_particles)
    

    grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h  #(n_particles, n_particles, n_parameters)

    
    v_svgd = -1 * torch.einsum('mn, mo -> no', K_XX, score_func_tensor) / n_particles + torch.mean(grad_K, dim=0) #(n_particles, n_parameters)

    

    
    if cfg.SVN.hessian_calc == "Full":  

        if cfg.SVN.block_diag_approx:
            alpha_list = []
            cg_maxiter = 50
            for i in range(n_particles):
                v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
                squared_kernel = K_XX**2
                H_part= hessians_tensor[i]*  squared_kernel[i][i] + torch.matmul(grad_K[i].T, grad_K[i])
                H_op_part = H_part.detach().cpu().numpy()
                alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=cg_maxiter)
                alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(device)
                alpha_list.append(alpha_part)
            alphas = torch.stack(alpha_list, dim=0).view(n_particles, -1)
            alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)

        else: 
            H1 = torch.einsum("xy, xz, xbd -> yzbd", K_XX, K_XX, hessians_tensor) #(n_particles, n_particles, n_parametes_per_model, n_parametes_per_model)
            H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
            
            #adds H2 values to diagonal of H1
            H1[range(n_particles), range(n_particles)] += H2

            N, _, D, _ = H1.shape  # Extract N and D from the shape of H
            H_reshaped = H1.permute(0, 2, 1, 3).reshape(N*D, N*D)  # Reshape to [N*D, N*D] ,  #(n_particles*n_parametes_per_model, n_particles*n_parametes_per_model)
            H =  H_reshaped / n_particles
            H_op = H.detach().cpu().numpy()
    
    elif cfg.SVN.hessian_calc == "Diag":  
        N = n_particles
        D = n_parameters
        
        if cfg.SVN.block_diag_approx:
            alpha_list = []
            cg_maxiter = 50     
            for i in range(n_particles):
                v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
                squared_kernel = K_XX**2
                #kernels_grads = torch.matmul(grad_K[i].T, grad_K[i])
                #H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: diag_hessian_matvec_block(x, squared_kernel[i][i],kernels_grads,hessians_tensor[i], device))
                H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: diag_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i],hessians_tensor[i], device))
                alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=cg_maxiter)
                alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(device)
                alpha_list.append(alpha_part)
            alphas = torch.stack(alpha_list, dim=0).view(n_particles, -1)
            alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        else:
            H1 = torch.einsum("xy, xz, xb -> yzb", K_XX, K_XX, hessians_tensor) #(n_particles, n_particles, n_parametes_per_model)
            H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
            
            #adds H2 values to diagonal of H1
            print(H1.shape)
            print(H2.shape)
            # Ensure H1 and H2 shapes are consistent for matmul
            H1_diag = H1[range(n_particles), range(n_particles)]  # Shape [n_particles, n_parameters_per_model]
            print("H1_diag shape:", H1_diag.shape)
            # Perform the matmul operation
            result = torch.matmul(H1_diag, H2)  # Shape [n_particles, n_parameters_per_model]
            print("Result shape:", result.shape)
            #H1[range(n_particles), range(n_particles)] = torch.matmul(H1[range(n_particles), range(n_particles)], H2)
            H1[range(n_particles), range(n_particles)] = result
            print("Updated H1 shape:", H1.shape)


            N, _, D, _ = H1.shape  # Extract N and D from the shape of H
            H_reshaped = H1.permute(0, 2, 1, 3).reshape(N*D, N*D)  # Reshape to [N*D, N*D] ,  #(n_particles*n_parametes_per_model, n_particles*n_parametes_per_model)
            H =  H_reshaped / n_particles
            H_op = H.detach().cpu().numpy()

    elif cfg.SVN.hessian_calc == "Kron":          
        # Create a linear operator that represents your Hessian vector product: Hx
        N = n_particles
        D = n_parameters

        if cfg.SVN.ll: 
            last_layer_params = 0
            eigenvalues = kron_decomposed.eigenvalues
            eigenvectors = kron_decomposed.eigenvectors
            for eigenvalues_layer, eigenvectors_layer in zip(eigenvalues, eigenvectors):
                if len(eigenvalues_layer) > 1:
                    V_K, V_Q = eigenvectors_layer
                    num_elements_K = V_K.shape[0]
                    num_elements_Q = V_Q.shape[0]
                    last_layer_params += num_elements_K * num_elements_Q
                else:
                    lambda_bias = eigenvalues_layer[0]
                    last_layer_params += lambda_bias.numel()
        
        if cfg.SVN.block_diag_approx:
            alpha_list = []
            cg_maxiter = 50     
            for i in range(n_particles):
                v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
                squared_kernel = K_XX**2
                if cfg.SVN.ll:
                    grad_K_i_last_layer = grad_K[i][:, -last_layer_params:]
                    H_op_part = scipy.sparse.linalg.LinearOperator((last_layer_params, last_layer_params), matvec=lambda x: kfac_hessian_matvec_block(x, squared_kernel[i][i],grad_K_i_last_layer, kron_list[i], device))
                    alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part[-last_layer_params:], maxiter=cg_maxiter)
                else:
                    H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: kfac_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i], kron_list[i], device))
                    alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=cg_maxiter)
                alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(device)
                alpha_list.append(alpha_part)
            alphas = torch.stack(alpha_list, dim=0).view(n_particles, -1)
            if cfg.SVN.ll:
                alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, last_layer_params)
                ll_v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, last_layer_params)
                v_svn = v_svgd
                v_svn[:, -last_layer_params:] = ll_v_svn

            else:
                alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, n_parameters)
                v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        else: 
            if cfg.SVN.ll:
                L = last_layer_params
                last_layer_grad_K = grad_K[:,:, -last_layer_params:]
                H2 = torch.einsum('xzi,xzj -> zij', last_layer_grad_K, last_layer_grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
                H_op = scipy.sparse.linalg.LinearOperator((N*L, N*L), matvec=lambda x: hessian_matvec(x, K_XX, kron_list, H2, last_layer_params, device))
            else:
                H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
                H_op = scipy.sparse.linalg.LinearOperator((N*D, N*D), matvec=lambda x: hessian_matvec(x, K_XX, kron_list, H2, n_parameters, device))


    solve_method = 'CG'
    # solve_method = 'Cholesky'

    if solve_method == 'CG' and not cfg.SVN.block_diag_approx:
        
        cg_maxiter = 50        


        if cfg.SVN.ll and cfg.SVN.hessian_calc=='Kron':
            last_layer_v_svgd = v_svgd[:, -last_layer_params:]
            ll_v_svgd_numpy = last_layer_v_svgd.detach().cpu().flatten().numpy()
            alphas, _ = scipy.sparse.linalg.cg(H_op, ll_v_svgd_numpy, maxiter=cg_maxiter)
            alphas = torch.tensor(alphas, dtype=torch.float32).to(device)                
        else:
            v_svgd_numpy = v_svgd.detach().cpu().flatten().numpy()
            alphas, _ = scipy.sparse.linalg.cg(H_op, v_svgd_numpy, maxiter=cg_maxiter)
            alphas = torch.tensor(alphas, dtype=torch.float32).reshape(n_particles, n_parameters).to(device)                

        if cfg.SVN.ll:
            alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, last_layer_params)
            ll_v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, last_layer_params)
            v_svn = v_svgd
            v_svn[:, -last_layer_params:] = ll_v_svn

        else:
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
        model.to(device)
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
    

