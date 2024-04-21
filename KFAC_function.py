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
    x_partition = x_partition.float()

    dim_K = V_K.size(0)
    dim_Q = V_Q.size(0)

    x_reshaped = x_partition.view(dim_Q, dim_K)

    # Reconstruct matrix representations from eigenvalues and eigenvectors
    matrix_Q = torch.matmul(V_Q, torch.matmul(torch.diag(lambda_Q), V_Q.T))
    matrix_K = torch.matmul(V_K, torch.matmul(torch.diag(lambda_K), V_K.T))
    kronecker_product = torch.kron(matrix_Q, matrix_K)
    result = torch.matmul(kronecker_product, x_partition)

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