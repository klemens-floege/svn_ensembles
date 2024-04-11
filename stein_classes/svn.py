import torch
import torch.autograd as autograd
import torch.utils.data as data_utils
import scipy
import numpy as np

from laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace

from stein_classes.stein_utils import calc_loss

def apply_kron_eigen_decomp(eigenvalues_layer, eigenvectors_layer, x_partition):
    """
    Applies the Kronecker product operation using eigen decomposition for a single layer.
    eigenvalues_layer: A tuple containing two tensors of eigenvalues for K and Q.
    eigenvectors_layer: A tuple containing two tensors of eigenvectors for K and Q.
    x_partition: The part of x corresponding to this layer.
    
    Returns the result of the operation for this layer.
    """
    # Unpack eigenvalues and eigenvectors
    lambda_K, lambda_Q = eigenvalues_layer
    V_K, V_Q = eigenvectors_layer
    
    # Assume x_partition is appropriately reshaped for this layer
    # The actual reshaping will depend on the dimensions of V_K and V_Q

    lambda_K = lambda_K.float()
    lambda_Q = lambda_Q.float()
    V_K = V_K.float()
    V_Q = V_Q.float()
    x_partition = x_partition.float()
    
    # Compute using eigen decompositions (this is a simplified conceptual approach)
    result = torch.kron(torch.matmul(V_K, torch.matmul(torch.diag(lambda_K), V_K.T)),
                        torch.matmul(V_Q, torch.matmul(torch.diag(lambda_Q), V_Q.T))) @ x_partition
    
    return result



def compute_KFACx(kron_decomposed, x):
    '''
    This function computes the matrix vector product A * x, for a matrix A which is given in kron.decomposed format
    '''
    
    KFACx_result = []
    x_index = 0  # Tracks our position in x as we partition it

    eigenvalues = kron_decomposed.eigenvalues
    eigenvectors = kron_decomposed.eigenvectors

    counter_parameters = 0
    for eigenvalues_layer, eigenvectors_layer in zip(eigenvalues, eigenvectors):
        
        #hande layer weights which are given as Kronecker factors
        if len(eigenvalues_layer) > 1:
            
            #compute number of elements in layer
            V_K, V_Q = eigenvectors_layer
            num_elements_K = V_K.shape[0]
            num_elements_Q = V_Q.shape[0]
            partition_size = num_elements_K * num_elements_Q

            # Extract the relevant partition of x and compute matmul with Kronecker product
            x_partition = x[x_index:x_index + partition_size]
            x_partition = torch.tensor(x_partition)
            x_partition.float()
            layer_result = apply_kron_eigen_decomp(eigenvalues_layer, eigenvectors_layer, x_partition)
            
        # Handle bias, which is represented an Eigendecomposition of full matrix
        else:
            
            lambda_bias = eigenvalues_layer[0]
            V_bias = eigenvectors_layer[0]
            partition_size = lambda_bias.numel()
      
            x_partition = torch.tensor(x[x_index:x_index + partition_size], dtype=torch.float32)
            V_bias = V_bias.clone().detach().to(dtype=torch.float32)
            lambda_bias = lambda_bias.clone().detach().to(dtype=torch.float32)

            layer_result = torch.matmul(V_bias, torch.matmul(torch.diag(lambda_bias), V_bias.T @ x_partition))
        
        counter_parameters += partition_size

        
        KFACx_result.append(layer_result.flatten())
        x_index += partition_size

    # Concatenate all results into a single vector
    result_tensor = torch.cat(KFACx_result)

    if counter_parameters != x.shape[0]:
        raise ValueError('The KFAC times vector multiplcation did not work. Total sum of opearted elements not euqal to size of vector')
    
    return result_tensor

def hessian_matvec(input, K_XX, kron_list, H2, n_parameters):

        n_particles = len(kron_list)
        result = torch.zeros_like(torch.tensor(input))


        # Now, use transformed_eigenvalues for further operations or aggregation
        # This involves broadcasting K_XX across the dimensions where it is needed
        weights = K_XX[:, :, None] * K_XX[:, None, :]  # Shape: [n_particles, n_particles, n_particles]
        

        for y in range(n_particles):
            for z in range(n_particles):
                for x in range(n_particles):
                    
                    current_parameter_vector = input[n_parameters*z: n_parameters*(z+1)]

                    result[n_parameters*y: n_parameters*(y+1)] += weights[x, y, z] * compute_KFACx(kron_list[x], current_parameter_vector)

                    if z == y:
                        if isinstance(current_parameter_vector, np.ndarray):
                            current_parameter_vector = torch.from_numpy(current_parameter_vector).float()
                        result[n_parameters*y: n_parameters*(y+1)] += torch.matmul(H2[y], current_parameter_vector)  


        return result.detach().numpy()

def apply_SVN(modellist, parameters, 
              batch, train_dataloader, kernel, device, cfg):
    
    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
    

    loss, log_prob = calc_loss(modellist, batch, train_dataloader, cfg, device)

    score_func = autograd.grad(log_prob.sum(), parameters)
    score_tensors = [t.view(-1) for t in score_func]  # Flatten
    score_func_tensor = torch.cat(score_tensors).view(n_particles, -1)  # (n_particles, n_parameters)


    inputs_squeezed = inputs.squeeze(1)  # Specify dimension to squeeze


    hessian_particle_loader = data_utils.DataLoader(
        data_utils.TensorDataset(inputs_squeezed, targets), 
        batch_size=cfg.SVN.hessian_calc_batch_size
        )
    
    
    hessians_list = []
    kron_list = []

    for i in range(n_particles):
        
        if cfg.SVN.hessian_calc == "Full":  
            laplace_particle_model = FullLaplace(modellist[i], likelihood='regression')
            laplace_particle_model.fit(hessian_particle_loader)
            Hessian = laplace_particle_model.posterior_precision
            hessians_list.append(Hessian)
            
        elif cfg.SVN.hessian_calc == "Diag":  
            laplace_particle_model = DiagLaplace(modellist[i], likelihood='regression')

            laplace_particle_model = LowRankLaplace(modellist[i], likelihood='regression')
            laplace_particle_model.fit(hessian_particle_loader)
            Hessian = laplace_particle_model.posterior_precision

            print(len(Hessian))
            print(len(Hessian[0]))
            print(Hessian[0][0].shape) #torch.Size([1021, 7])
            print(Hessian[0][1].shape) #torch.Size([7])
            print(Hessian[1][0].shape) #torch.Size([])
            print(Hessian[1][1].shape) #torch.Size([])

        elif cfg.SVN.hessian_calc == "Kron":  
            laplace_particle_model = KronLaplace(modellist[i], likelihood='regression')
            laplace_particle_model.fit(hessian_particle_loader)
            kron_decomposed = laplace_particle_model.posterior_precision


            kron_list.append(kron_decomposed)

        elif cfg.SVN.hessian_calc == "LowRank":  
            laplace_particle_model = LowRankLaplace(modellist[i], likelihood='regression')
            laplace_particle_model.fit(hessian_particle_loader)
            Hessian = laplace_particle_model.posterior_precision[0]

            print(Hessian)
            #print(Hessian.shape)
            hessians_list.append(Hessian)
            
        else: 
            ValueError("This is type of Hessian calculation is not yet implemented")
        
        
    if cfg.SVN.hessian_calc == "Full":  
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
    K_XX = torch.exp(-squared_distances / h) #(n_particles, n_particles)
    

    grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h  #(n_particles, n_particles, n_parameters)
    
    v_svgd = -1 * torch.einsum('mn, mo -> no', K_XX, score_func_tensor) / n_particles + torch.mean(grad_K, dim=0) #(n_particles, n_parameters)

    H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)

    
    if cfg.SVN.hessian_calc == "Full":  
        H1 = torch.einsum("xy, xz, xbd -> yzbd", K_XX, K_XX, hessians_tensor) #(n_particles, n_particles, n_parametes_per_model, n_parametes_per_model)
        
        #adds H2 values to diagonal of H1
        H1[range(n_particles), range(n_particles)] += H2

        N, _, D, _ = H1.shape  # Extract N and D from the shape of H
        H_reshaped = H1.permute(0, 2, 1, 3).reshape(N*D, N*D)  # Reshape to [N*D, N*D] ,  #(n_particles*n_parametes_per_model, n_particles*n_parametes_per_model)
        H =  H_reshaped / n_particles
        H_op = H.detach().cpu().numpy()

    elif cfg.SVN.hessian_calc == "Kron":          
        # Create a linear operator that represents your Hessian vector product: Hx
        N = n_particles
        D = n_parameters
        H_op = scipy.sparse.linalg.LinearOperator((N*D, N*D), matvec=lambda x: hessian_matvec(x, K_XX, kron_list, H2, n_parameters))


    solve_method = 'CG'
    # solve_method = 'Cholesky'

    if solve_method == 'CG':
        
        cg_maxiter = 50        

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
    

