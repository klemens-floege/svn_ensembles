import torch
import scipy
import numpy as np


def calc_loss(modellist, batch,
              train_dataloader, cfg, device):

    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)

    dim_problem = targets.shape[1]
    
    pred_list = []

    for i in range(n_particles):
        pred_list.append(modellist[i].forward(inputs))

    pred = torch.cat(pred_list, dim=0)
    pred_reshaped = pred.view(n_particles, -1, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]

    # Mean prediction
    ensemble_pred = torch.mean(pred_reshaped, dim=0) 

    mse_loss = (ensemble_pred - targets) ** 2
    loss = mse_loss
    pred_dist_std = cfg.SVN.red_dist_std
    ll = -loss*len(train_dataloader) / pred_dist_std ** 2
    log_prob = ll

    return loss, log_prob




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

    dim_K = V_K.size(0)
    dim_Q = V_Q.size(0)

    x_reshaped = x_partition.view(dim_Q, dim_K)

    # Step 1: Multiply with V_Q and D_Q
    step1 = torch.matmul(V_Q, torch.matmul(torch.diag(lambda_Q), V_Q.T))

    # Step 2: Multiply the result of step 1 with x_reshaped
    # The resulting matrix has the same shape as x_reshaped
    intermediate_result = torch.matmul(step1, x_reshaped)

    # Step 3: Multiply with V_K and D_K
    # Before this multiplication, transpose the intermediate result to match dimensions for multiplication
    step2 = torch.matmul(V_K, torch.matmul(torch.diag(lambda_K), V_K.T))

    # Final step: Multiply the result of step 3 with the transposed intermediate result
    # Then, 'flatten' the result back into a vector form to match the original x_partition's shape
    result = torch.matmul(step2, intermediate_result.T).flatten()

    
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
                current_parameter_vector = input[n_parameters*z: n_parameters*(z+1)]
                for x in range(n_particles):
                    
                    result[n_parameters*y: n_parameters*(y+1)] += weights[x, y, z] * compute_KFACx(kron_list[x], current_parameter_vector)

                #TODO: double check intendation:
                if z == y:
                    if isinstance(current_parameter_vector, np.ndarray):
                        current_parameter_vector = torch.from_numpy(current_parameter_vector).float()
                    result[n_parameters*y: n_parameters*(y+1)] += torch.matmul(H2[y], current_parameter_vector)  


        return result.detach().numpy()