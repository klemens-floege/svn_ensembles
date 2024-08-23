import torch
import torch.autograd as autograd
import torch.utils.data as data_utils
import scipy
import numpy as np

from laplace import  DiagLaplace
from laplace.curvature import AsdlGGN, AsdlEF, GGNInterface
from laplace import Laplace

from stein_classes.loss import calc_loss
from stein_classes.stein_utils import hessian_matvec, kfac_hessian_matvec_block, diag_hessian_matvec_block


from SVN.hessian_approximation import HessianApproximation

class DiagHessian(HessianApproximation):

    def compute_hessians(self):
        
        #return torch.ones((self.n_particles, self.n_parameters))
        #if self.cfg.SVN.use_adam_hessian and 1 < self.step:
        if self.cfg.optimizer.type in ["Adam", "AdamW"] and self.cfg.SVN.use_adam_hessian: 
            if 1 < self.step:
                return self.get_adam_hessians()
            else:
                return torch.ones((self.n_particles, self.n_parameters))
        elif self.cfg.optimizer.type in ["IVON"]:
            return self.get_ivon_hessians()
        else: 
            if 0 < self.step:
                return self.last_hessian
            else:
                self.last_hessian = self.get_laplace_lib_hessians()
                return self.get_laplace_lib_hessians()
            
        

    def compute_kernel(self, hessians_tensor, parameter_tensors):

        if self.cfg.SVN.use_curvature_kernel == "use_curvature":
            M = torch.mean(hessians_tensor, axis=0)
        else:
            M=None

        ensemble_parameters_tensor = torch.cat(parameter_tensors).view(self.n_particles, -1)  # Concatenate and reshape

        h = self.cfg.experiment.kernel_width        
        displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
        
        if M is not None:
            ensemble_parameters_tensor = M * ensemble_parameters_tensor
            displacement_tensor = displacement_tensor * M
            
        squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
        
        K_XX = torch.exp(-squared_distances / h) #(n_particles, n_particles)
        grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h  #(n_particles, n_particles, n_parameters)

        return K_XX, grad_K


    def solve_linear_system(self, hessians_tensor, K_XX, grad_K, v_svgd):
        return None
    
    def solve_block_linear_system(self,hessians_tensor, K_XX, grad_K, v_svgd):
        use_analytic=True

        if use_analytic:
            return self.solve_analytic_block_linear_system(hessians_tensor, K_XX, grad_K, v_svgd)
        else:
            N, D = v_svgd.shape
            alpha_list = []
            cg_maxiter = 50     
            for i in range(self.n_particles):
                v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
                squared_kernel = K_XX**2
                H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: diag_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i],hessians_tensor[i], self.device))
                alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=cg_maxiter)
                alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(self.device)
                alpha_list.append(alpha_part)
            alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
            return v_svn

    #TODO: creates numerical instabilities why?
    def solve_analytic_block_linear_system(self,hessians_tensor, K_XX, grad_K, v_svgd):
        # Assuming squared_kernel, diag_hessian, and grad_K_i are given and on the correct device
        squared_kernel = K_XX**2
        epsilon = 1e-6  # Small value for numerical stability
        alpha_list = []
        for i in range(self.n_particles):
            squared_kernel_i = squared_kernel[i][i].clone().detach().float().to(self.device)
            grad_K_i = grad_K[i][i].clone().detach().float().to(self.device)
            diag_hessian = hessians_tensor[i].clone().detach().float().to(self.device)

            '''D = squared_kernel_i * diag_hessian # Compute the diagonal matrix D
            D_inv = 1.0 / (D + epsilon)  # Adding epsilon directly to D
            D_inv = torch.diag(D_inv) # Reshape D_inv and grad_K_i for matrix operations
            grad_K_i = grad_K_i.view(-1, 1)
            u_T_D_inv_u = (grad_K_i.t() @ D_inv @ grad_K_i).item() # Compute u^T D^-1 u
            D_inv_u = D_inv @ grad_K_i # Compute the outer product D^-1 u u^T D^-1
            outer_product = D_inv_u @ D_inv_u.t()
            #A_inv = D_inv - outer_product / (1 + u_T_D_inv_u) # Compute the inverse using the Sherman-Morrison formula
            A_inv = D_inv - outer_product / (1 + u_T_D_inv_u + epsilon)  # Add epsilon to the denominator for stability
            alpha_part = A_inv @ v_svgd[i]
            alpha_list.append(alpha_part)'''


            D = squared_kernel_i * diag_hessian # Compute the diagonal matrix D
            D_inv = 1.0 / (D + epsilon)  # Adding epsilon directly to D
            u_T_D_inv_u = torch.sum(D_inv * grad_K_i**2)  # Computing u^T D^-1 u, which is a scalar
            D_inv_u = D_inv * grad_K_i # Compute the outer product D^-1 u u^T D^-1

            
            scaling_factor = 1 / (1 + u_T_D_inv_u + epsilon)
            term1= D_inv * v_svgd[i]
            term2 = torch.sum(D_inv_u * v_svgd[i])
            alpha_part = term1 - (scaling_factor * term2) * D_inv_u
            #outer_product = D_inv_u @ D_inv_u.t()
            #A_inv = D_inv - outer_product / (1 + u_T_D_inv_u + epsilon)  # Add epsilon to the denominator for stability
            #alpha_part = A_inv @ v_svgd[i]
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
        alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
        v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        

        return v_svn
    

    def get_laplace_lib_hessians(self):
        hessian_list = []
        for i in range(self.n_particles):

            if self.cfg.SVN.classification_likelihood:
                laplace_particle_model = DiagLaplace(self.modellist[i], 
                                                        likelihood=self.cfg.task.task_type)
            else:
                laplace_particle_model = DiagLaplace(self.modellist[i], 
                                                        likelihood='regression')
            laplace_particle_model.fit(self.hessian_particle_loader)
            particle_hessian = laplace_particle_model.posterior_precision
            hessian_list.append(particle_hessian)
            
        hessians_tensor = torch.cat(hessian_list, dim=0)
        hessians_tensor = hessians_tensor.reshape(self.n_particles, self.n_parameters) #(n_particles, n_parameters)
        return hessians_tensor

    def get_adam_hessians(self):


        second_moments = [
            state['exp_avg_sq'].flatten()
            for state in self.optimizer.state.values() if 'exp_avg_sq' in state
        ]

        if not second_moments:
            raise ValueError("No second moments found in the optimizer state.")
        
        # Print the second moments list
        #for idx, moments in enumerate(second_moments):
          #  print(f"Second moments for parameter {idx}: {moments}")
        # Stack the flattened second moments to form a tensor
        second_moments_tensor = torch.cat(second_moments, dim=0)
        

        beta2 = self.optimizer.param_groups[0]['betas'][1]
        t = self.step

        # Calculate v_t with in-place operation
        v_hat_tensor = second_moments_tensor * (1 - beta2 ** t)
        
        #Hacky
        #sqrt_v_hat_tensor = torch.sqrt(v_hat_tensor)  # Adding epsilon for numerical stability
        #hessians_tensor = self.cfg.experiment.batch_size * sqrt_v_hat_tensor.view(self.n_particles, -1)

        # Reshape to the correct dimensions
        sqrt_batch_size = torch.sqrt(torch.tensor(self.cfg.experiment.batch_size, dtype=torch.float32))

        hessians_tensor = self.cfg.experiment.batch_size * v_hat_tensor.view(self.n_particles, -1)
        #hessians_tensor = (1/sqrt_batch_size) * v_hat_tensor.view(self.n_particles, -1)
        #hessians_tensor = sqrt_batch_size * v_hat_tensor.view(self.n_particles, -1)

        scale_factor = 1e-5  # or another small value depending on the magnitude
        #hessians_tensor = scale_factor * hessians_tensor
        hessians_tensor = (hessians_tensor - hessians_tensor.mean()) / hessians_tensor.std()

        
        #print('hesisans: ', hessians_tensor[:5])
        #print(hessians_tensor.shape)
        
        return hessians_tensor

    def get_ivon_hessians(self):

        hessians = []
        #print(self.optimizer.param_groups)
        #print(len(self.optimizer.param_groups))
        """for group in self.optimizer.param_groups:
            hessian = group["hess"]
            print('particle hessian shape',  hessian.shape)
            hessians.append(hessian)"""
        #print(self.optimizer.param_groups[0].keys())

        #flat_hessians_tensor = torch.cat(hessians, 0)
        flat_hessians_tensor = self.optimizer.param_groups[0]["hess"]
        hessians_tensor = self.cfg.experiment.batch_size * flat_hessians_tensor.view(self.n_particles, -1)

        

        #print('hesisans: ', hessians_tensor[:5])
        #print('hesisans: ', hessians_tensor.shape)
        
        return hessians_tensor
        
        
            
        
        
        
        
