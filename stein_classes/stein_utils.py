import scipy
import numpy as np

import torch
 




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
    x_partition = x_partition.float() #flat vector 
    

    dim_K = V_K.size(0)
    dim_Q = V_Q.size(0)

    x_matrix = x_partition.view(dim_Q, dim_K)

    # Reconstruct matrix representations from eigenvalues and eigenvectors
    #matrix_K = torch.matmul(V_K, torch.matmul(torch.diag(lambda_K), V_K.T)) #Large Factor
    #matrix_Q = torch.matmul(V_Q, torch.matmul(torch.diag(lambda_Q), V_Q.T)) #Small Factor
    
    #Use vectorization trick and use that K, V are symmetric
    #result = torch.matmul(torch.matmul(matrix_Q, x_matrix), matrix_K).flatten()

    #Likely Better
    #1. Tranform X to Eigenspace of Q and K 
    x_prime = torch.matmul(torch.matmul(V_Q.T, x_matrix), V_K)
    #2. Scale by Eigenvalues
    x_double_prime = (lambda_Q.unsqueeze(1) * x_prime) * lambda_K.unsqueeze(0)
    # Step 3: Transform x_double_prime back from the eigenspaces
    result = torch.matmul(torch.matmul(V_Q, x_double_prime), V_K.T)
    

    #step1 = torch.matmul(V_Q, torch.matmul(torch.diag(lambda_Q), V_Q.T))
    #intermediate_result = torch.matmul(step1, x_reshaped)
    #step2 = torch.matmul(V_K, torch.matmul(torch.diag(lambda_K), V_K.T))
    #result = torch.matmul(step2, intermediate_result.T).flatten()

    
    # Compute using eigen decompositions (this is a simplified conceptual approach)
    #result = torch.kron(torch.matmul(V_K, torch.matmul(torch.diag(lambda_K), V_K.T)),
    #                    torch.matmul(V_Q, torch.matmul(torch.diag(lambda_Q), V_Q.T))) @ x_partition
    
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
            x_partition = x_partition
            x_partition.float()
            layer_result = apply_kron_eigen_decomp(eigenvalues_layer, eigenvectors_layer, x_partition)
            
        # Handle bias, which is represented an Eigendecomposition of full matrix
        else:
            
            lambda_bias = eigenvalues_layer[0]
            V_bias = eigenvectors_layer[0]
            partition_size = lambda_bias.numel()
      
            x_partition = x[x_index:x_index + partition_size]
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



def hessian_matvec(input, K_XX, kron_list, H2, n_parameters, device):

        n_particles = len(kron_list)
        result = torch.zeros_like(torch.tensor(input)).to(device)

        input = torch.tensor(input).float().to(device)
        grad_K = torch.tensor(grad_K).float().to(device)
        K_XX = torch.tensor(K_XX).float().to(device)
        H2 = torch.tensor(H2).float().to(device)

        # Now, use transformed_eigenvalues for further operations or aggregation
        # This involves broadcasting K_XX across the dimensions where it is needed
        weights = K_XX[:, :, None] * K_XX[:, None, :]  # Shape: [n_particles, n_particles, n_particles]
        
        
        for z in range(n_particles):
            current_parameter_vector = input[n_parameters*z: n_parameters*(z+1)]

            list_of_KFAC_hess = []
            for x in range(n_particles):
                list_of_KFAC_hess.append(compute_KFACx(kron_list[x], current_parameter_vector))
            
            #This was slower
            #list_of_KFAC_hess = parallel_kfac(kron_list, current_parameter_vector, n_particles)

            for y in range(n_particles):
                #can be implmented faster
                for x in range(n_particles):
                    result[n_parameters*y: n_parameters*(y+1)] += weights[x, y, z] * list_of_KFAC_hess[x]

                #TODO: double check intendation:
                if z == y:
                    if isinstance(current_parameter_vector, np.ndarray):
                        current_parameter_vector = torch.from_numpy(current_parameter_vector).float()
                    result[n_parameters*y: n_parameters*(y+1)] += torch.matmul(H2[y], current_parameter_vector)  


        return result.detach().numpy()

 

#Blockwise Hessian Matrix Block for KFAC computation
def kfac_hessian_matvec_block(input, squared_kernel, kernels_grads, kron, device):


        input = torch.tensor(input).float().to(device)
        squared_kernel = torch.tensor(squared_kernel).float().to(device)
        kernels_grads = torch.tensor(kernels_grads).float().to(device)
        

        kernel_grads_vector = torch.matmul(kernels_grads, input)
        kernel_weght_param_vector = squared_kernel * input
        hess_vector = compute_KFACx(kron, kernel_weght_param_vector)
        
        update = hess_vector + kernel_grads_vector


        return update.detach().cpu().numpy()

#Blockwise Hessian Matrix Block for KFAC computation
def diag_hessian_matvec_block(input, squared_kernel, kernels_grads, diag_hessian, device):

        
        input = torch.tensor(input).float().to(device)
        squared_kernel = torch.tensor(squared_kernel).float().to(device)
        kernels_grads = torch.tensor(kernels_grads).float().to(device)
        diag_hessian = torch.tensor(diag_hessian).float().to(device)

        kernel_grads_vector = torch.matmul(kernels_grads, input)
        kernel_weght_param_vector = squared_kernel * input
        hess_vector = diag_hessian * kernel_weght_param_vector
        
        update = hess_vector + kernel_grads_vector


        return update.detach().cpu().numpy()