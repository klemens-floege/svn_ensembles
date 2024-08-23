
import torch
import scipy
import numpy as np

from laplace import KronLaplace
from laplace.curvature import AsdlGGN, AsdlEF, GGNInterface
from laplace import Laplace

import jax
import jax.numpy as jnp
from functools import partial

from stein_classes.loss import calc_loss
from stein_classes.stein_utils import hessian_matvec, kfac_hessian_matvec_block, diag_hessian_matvec_block


from SVN.hessian_approximation import HessianApproximation
from SVN.kfac_jax_utils import jitted_compute_bias_KFACx, jitted_compute_kron_KFACx

class KronHessian(HessianApproximation):

    def __init__(self, model, cfg, device, optimizer):
        # Pass the required arguments to the parent class
        super().__init__(model, cfg, device, optimizer)
        
        
        self.kron_list = None
        self.inverted_kron_list = None

    def compute_hessians(self):
        if 0 < self.step:
            return self.kron_list 
        else:
            hessian_list = []
            for i in range(self.n_particles):

                if self.cfg.SVN.ll:
                    laplace_particle_model = Laplace(self.modellist[i], 'regression')
                elif self.cfg.SVN.classification_likelihood: 
                    laplace_particle_model = KronLaplace(self.modellist[i], 
                                                                likelihood=self.cfg.task.task_type)
                else:
                    laplace_particle_model = KronLaplace(self.modellist[i], 
                                                    likelihood='regression'                                                 
                                                    )
                laplace_particle_model.fit(self.hessian_particle_loader)
                kron_decomposed = laplace_particle_model.posterior_precision
                hessian_list.append(kron_decomposed)
                
            self.kron_list = hessian_list
            self.inverted_kron_list = self.compute_invert_kron(self.kron_list)
            return hessian_list
        
    def compute_invert_kron(self, kron_list):
        return None

    
    def compute_kernel(self, hessians_tensor, parameter_tensors):

        if self.cfg.SVN.use_curvature_kernel == "use_curvature":
             raise NotImplementedError("Anisotropic Gaussian Curvature Kernel is not implemented yet for KFAC Hessian.")
        
        ensemble_parameters_tensor = torch.cat(parameter_tensors).view(self.n_particles, -1)  # Concatenate and reshape
        h = self.cfg.experiment.kernel_width        
        displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
        squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
        K_XX = torch.exp(-squared_distances / h) #(n_particles, n_particles)
        grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h  #(n_particles, n_particles, n_parameters)
        return K_XX, grad_K
    
    
    def split_bias_kron(self, eigenvalues, block_types):
        kron_eigenvalues = []
        bias_eigenvalues = []
        
        for eigenvalue, block_type in zip(eigenvalues, block_types):
            if block_type == 0:
                kron_eigenvalues.append(eigenvalue)
            elif block_type == 1:
                bias_eigenvalues.append(eigenvalue)
        
        return kron_eigenvalues, bias_eigenvalues
    

    def pad_bias_array(self, arr, target_length):
        return jnp.pad(arr, (0, target_length - arr.shape[0]), 'constant', constant_values=0)
    
    def pad_to_shape(self, x, target_shape):
        pad_width = [(0, target_shape[i] - x.shape[i]) for i in range(len(x.shape))]
        return jnp.pad(x, pad_width, mode='constant')


    def compute_KFAC_x(self, kron_decomposed, input, inverse=False):

        eigenvalues = kron_decomposed.eigenvalues
        eigenvectors = kron_decomposed.eigenvectors

        # Convert PyTorch tensors to JAX arrays
        eigenvalues = [[jnp.array(ev.detach().cpu().numpy()) for ev in layer] for layer in eigenvalues]
        eigenvectors = [[jnp.array(ev.detach().cpu().numpy()) for ev in layer] for layer in eigenvectors]
        #input = jnp.array(input.detach().cpu().numpy())
        if isinstance(input, torch.Tensor):
            input = jnp.array(input.detach().cpu().numpy())
        elif isinstance(input, np.ndarray):
            input = jnp.array(input)
        elif not isinstance(input, jnp.ndarray):
            raise TypeError("Input must be a PyTorch tensor, NumPy array, or JAX array")

        # Invert eigenvalues if inverse flag is True, in order to compute A_inverse * x 
        if inverse:
            epsilon=10e-6
            eigenvalues = [[1.0 / (ev + epsilon) for ev in layer] for layer in eigenvalues]


        # Determine block sizes
        partition_sizes = jnp.array([
            eigenvectors_layer[0].shape[0] * eigenvectors_layer[1].shape[0] if len(eigenvalues_layer) > 1 else len(eigenvalues_layer[0])
            for eigenvalues_layer, eigenvectors_layer in zip(eigenvalues, eigenvectors)
        ])

        # Determine block types (0: Kronecker factor, 1: Bias)
        block_types = jnp.arange(len(eigenvalues)) % 2

        kron_eigenvalues, bias_eigenvalues = self.split_bias_kron(eigenvalues, block_types)
        kron_eigenvectors, bias_eigenvectors = self.split_bias_kron(eigenvectors, block_types)

        assert len(kron_eigenvalues) == len(bias_eigenvalues)


        #TODO: make nicer, this is max bias block
        max_bias_block = 0
        for i in eigenvalues:
            if len(i) == 1:
                if i[0].shape[0] > max_bias_block:
                    max_bias_block = i[0].shape[0]
        
        #TODO: replace with max layer sizes
        max_kron_block = 50

        max_kron_block = jnp.array(max_kron_block)
        max_bias_block = jnp.array(max_bias_block)
        

        padded_bias_eigenvalues = jnp.array([
            self.pad_bias_array(sublist[0], max_bias_block) for sublist in bias_eigenvalues #(num_bias_blocks, max_bias_block)
            ])
        padded_bias_eigenvectors = jnp.array([
            self.pad_to_shape(sublist[0], (max_bias_block, max_bias_block)) for sublist in bias_eigenvectors #(num_bias_blocks, max_bias_block, max_bias_block)
            ])
        padded_kron_eigenvalues = jnp.array([
                jnp.array([self.pad_bias_array(sublist[0], max_kron_block), self.pad_bias_array(sublist[1], max_kron_block)]) for sublist in kron_eigenvalues #(num_bias_blocks, 2, max_kron_block)
            ])
        padded_kron_eigenvectors = jnp.array([
                jnp.array([self.pad_to_shape(sublist[0], (max_kron_block, max_kron_block)), self.pad_to_shape(sublist[1], (max_kron_block, max_kron_block))]) for sublist in kron_eigenvectors #(num_bias_blocks, 2, max_kron_block, max_kron_block)
            ])


        mask = (block_types == 1)

        split_indices = jnp.cumsum(partition_sizes)[:-1]
        split_input = jnp.split(input, split_indices)


        kron_input = split_input[0::2]
        bias_input = split_input[1::2]


        padded_kron_input = jnp.array([jnp.pad(arr, (0, max_kron_block**2 - arr.shape[0]), mode='constant') for arr in kron_input])
        padded_bias_input = jnp.array([jnp.pad(arr, (0, max_bias_block - arr.shape[0]), mode='constant') for arr in bias_input])


        kron_padded_result = jitted_compute_kron_KFACx(padded_kron_eigenvalues, padded_kron_eigenvectors, padded_kron_input)
        kron_padded_result = kron_padded_result.reshape((len(padded_kron_eigenvalues)), -1)
        
        bias_padded_result = jitted_compute_bias_KFACx(padded_bias_eigenvalues, padded_bias_eigenvectors, padded_bias_input)
        bias_padded_result = bias_padded_result.reshape((len(padded_bias_eigenvalues)), -1)

    
        kron_result_list = [part[:size] for part, size in zip(kron_padded_result, partition_sizes[~mask])]
        bias_result_list = [part[:size] for part, size in zip(bias_padded_result, partition_sizes[mask])]
        
        result_list = []
        for kron, bias in zip(kron_result_list, bias_result_list):
            result_list.append(kron)
            result_list.append(bias)

        result = jnp.concatenate(result_list)


        return jnp.atleast_1d(result)
    
    

    def analytic_kfac_system(self, kron_decomposed, squared_kernel_i, grad_K_i, v_svgd_i):


        squared_kernel_i = jnp.array(squared_kernel_i)
        grad_K_i = jnp.array(grad_K_i)
        v_svgd_i = jnp.array(v_svgd_i.detach().numpy())

        A_inverse_v_svgd_i = self.compute_KFAC_x(kron_decomposed, v_svgd_i, inverse=True)
        A_inverse_grad_K_i = self.compute_KFAC_x(kron_decomposed, grad_K_i, inverse=True)

        #term1 = torch.tensor(self.compute_KFAC_x(kron_decomposed, v_svgd_i, inverse=True)) * (1/squared_kernel_i)
        #grad_K_i_scaled = self.compute_KFAC_x(kron_decomposed, grad_K_i, inverse=True) * (1/squared_kernel_i)
        term1 = A_inverse_v_svgd_i * (1/squared_kernel_i)
        grad_K_i_scaled = A_inverse_grad_K_i * (1/squared_kernel_i)
        scaling_factor = 1 + jnp.dot(grad_K_i, grad_K_i_scaled)
        
        A_inv_u = A_inverse_grad_K_i *  (1/squared_kernel_i)
        #A_inv_u_ut_A_inv = jnp.dot(A_inv_u.T, A_inv_u)
        A_inv_u_ut_A_inv = jnp.outer(A_inv_u, A_inv_u)
        term2 = scaling_factor * jnp.dot(A_inv_u_ut_A_inv, v_svgd_i)
        alpha_part = term1 + term2
        
        # Convert the final result to a PyTorch tensor
        #alpha_part_tensor = torch.from_numpy(jnp.array(alpha_part))
        # Convert the JAX array to a NumPy array, then to a PyTorch tensor
        alpha_part_np = np.array(alpha_part)
        alpha_part_tensor = torch.from_numpy(alpha_part_np)
        return alpha_part_tensor
        

        """term1 = self.jax_compute_KFACx(kron_decomposed_inverse, v_svgd_i) * (1/squared_kernel_i)
        grad_K_i_scaled = self.jax_compute_KFACx(kron_decomposed_inverse, grad_K_i) * (1/squared_kernel_i)
        scaling_factor = 1 + jnp.dot(grad_K_i, grad_K_i_scaled)
        A_inv_u = self.jax_compute_KFACx(kron_decomposed_inverse, grad_K_i) *  (1/squared_kernel_i)
        A_inv_u_ut_A_inv = jnp.dot(A_inv_u.T, A_inv_u)
        term2 = scaling_factor * jnp.dot(A_inv_u_ut_A_inv, v_svgd_i)
        alpha_part = term1 + term2
        return alpha_part"""
    

    def solve_analytic_block_linear_system(self, kron_hessians, K_XX, grad_K, v_svgd):
        # Assuming squared_kernel, diag_hessian, and grad_K_i are given and on the correct device
        squared_kernel = K_XX**2
        epsilon = 1e-6  # Small value for numerical stability
        alpha_list = []        

        for i in range(self.n_particles):
            squared_kernel_i = squared_kernel[i][i].clone().detach().float().to(self.device)
            grad_K_i = grad_K[i][i].clone().detach().float().to(self.device)
            v_svgd_i = v_svgd[i]
            alpha_part = self.analytic_kfac_system(kron_hessians[i], squared_kernel_i, grad_K_i, v_svgd_i)
            
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
        alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
        v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        

        return v_svn

    def solve_linear_system(self, kron_hessians, K_XX, grad_K, v_svgd):
        N = self.n_particles
        D = self.n_parameters

        if self.cfg.SVN.ll:
            last_layer_params = self.calculate_last_layer_params(self, kron_hessians)
            L = last_layer_params
            last_layer_grad_K = grad_K[:,:, -last_layer_params:]
            H2 = torch.einsum('xzi,xzj -> zij', last_layer_grad_K, last_layer_grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
            H_op = scipy.sparse.linalg.LinearOperator((N*L, N*L), matvec=lambda x: hessian_matvec(x, K_XX, kron_hessians, H2, last_layer_params, self.device))
            last_layer_v_svgd = v_svgd[:, -last_layer_params:]
            ll_v_svgd_numpy = last_layer_v_svgd.detach().cpu().flatten().numpy()
            alphas, _ = scipy.sparse.linalg.cg(H_op, ll_v_svgd_numpy, maxiter=self.cg_maxiter)
            alphas = torch.tensor(alphas, dtype=torch.float32).to(self.device)      
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, last_layer_params)
            ll_v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, last_layer_params)
            v_svn = v_svgd
            v_svn[:, -last_layer_params:] = ll_v_svn         
        else:
            H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
            H_op = scipy.sparse.linalg.LinearOperator((N*D, N*D), matvec=lambda x: self.hessian_matvec(x, K_XX, kron_hessians, H2, self.n_parameters, self.device))
            v_svgd_numpy = v_svgd.detach().cpu().flatten().numpy()
            alphas, _ = scipy.sparse.linalg.cg(H_op, v_svgd_numpy, maxiter=self.cg_maxiter)
            alphas = torch.tensor(alphas, dtype=torch.float32).reshape(self.n_particles, self.n_parameters).to(self.device)                
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
       
        return v_svn

    #Blockwise Hessian Matrix Block for KFAC computation
    def kfac_hessian_matvec_block(self, input, squared_kernel, grad_K_i, kron, device):


            input = torch.tensor(input).float().to(device)
            squared_kernel = torch.tensor(squared_kernel).float().to(device)
            #kernels_grads = torch.tensor(kernels_grads).float().to(device)
            grad_K_i = torch.tensor(grad_K_i).float().to(device)
            

            #kernel_grads_vector = torch.matmul(kernels_grads, input)
            kernel_grads_vector = torch.matmul(torch.matmul(grad_K_i.T, grad_K_i), input)
            kernel_weght_param_vector = squared_kernel * input
            hess_vector = self.compute_KFAC_x(kron, kernel_weght_param_vector)
            hess_vector_np = np.array(hess_vector)
            hess_vectortensor = torch.from_numpy(hess_vector_np)
            # Check for NaN values
            if torch.isnan(hess_vectortensor).any():
                print(hess_vectortensor)
            
            update = hess_vectortensor + kernel_grads_vector


            return update.detach().cpu().numpy()
    

    def hessian_matvec(self, input, K_XX, kron_list, H2, n_parameters, device):

        n_particles = len(kron_list)
        result = torch.zeros_like(torch.tensor(input)).to(device)

        input = torch.tensor(input).clone().detach().float().to(device)
        #grad_K = torch.tensor(grad_K).float().to(device)
        K_XX = torch.tensor(K_XX).clone().detach().float().to(device)
        H2 = torch.tensor(H2).clone().detach().float().to(device)

        # Now, use transformed_eigenvalues for further operations or aggregation
        # This involves broadcasting K_XX across the dimensions where it is needed
        weights = K_XX[:, :, None] * K_XX[:, None, :]  # Shape: [n_particles, n_particles, n_particles]
        
        
        for z in range(n_particles):
            current_parameter_vector = input[n_parameters*z: n_parameters*(z+1)]

            list_of_KFAC_hess = []
            for x in range(n_particles):
                hess_vector=self.compute_KFAC_x(kron_list[x], current_parameter_vector)
                hess_vector_np = np.array(hess_vector)
                hess_vectortensor = torch.from_numpy(hess_vector_np)
                list_of_KFAC_hess.append(hess_vectortensor)
            
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
 
    
    def solve_block_linear_system(self,kron_hessians, K_XX, grad_K, v_svgd):
        analytic = False
        if analytic:
            return self.solve_analytic_block_linear_system(kron_hessians, K_XX, grad_K, v_svgd)

        N = self.n_particles
        D = self.n_parameters

        if self.cfg.SVN.ll:
            raise ValueError('Use new function')
       
        alpha_list = []
        for i in range(self.n_particles):
            v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
            squared_kernel = K_XX**2
            H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: self.kfac_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i], self.kron_list[i], self.device))
            alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=self.cg_maxiter)
            alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(self.device)
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
        alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
        v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        return v_svn
    
    def solve_last_layer_block_linear_system(self,kron_hessians, K_XX, grad_K, v_svgd):
        N = self.n_particles
        D = self.n_parameters

        last_layer_params = self.calculate_last_layer_params(self, kron_hessians)
        alpha_list = []
        for i in range(self.n_particles):
            v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
            squared_kernel = K_XX**2
            grad_K_i_last_layer = grad_K[i][:, -last_layer_params:]
            H_op_part = scipy.sparse.linalg.LinearOperator((last_layer_params, last_layer_params), matvec=lambda x: self.kfac_hessian_matvec_block(x, squared_kernel[i][i],grad_K_i_last_layer, kron_list[i], device))
            alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part[-last_layer_params:], maxiter=self.cg_maxiter)
            alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(self.device)
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
        alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, last_layer_params)
        ll_v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, last_layer_params)
        v_svn = v_svgd
        v_svn[:, -last_layer_params:] = ll_v_svn
    
    def calculate_last_layer_params(self, kron_hessians):
        last_layer_params = 0
        eigenvalues = kron_hessians.eigenvalues
        eigenvectors = kron_hessians.eigenvectors
        for eigenvalues_layer, eigenvectors_layer in zip(eigenvalues, eigenvectors):
            if len(eigenvalues_layer) > 1:
                V_K, V_Q = eigenvectors_layer
                num_elements_K = V_K.shape[0]
                num_elements_Q = V_Q.shape[0]
                last_layer_params += num_elements_K * num_elements_Q
            else:
                lambda_bias = eigenvalues_layer[0]
                last_layer_params += lambda_bias.numel()
        return last_layer_params