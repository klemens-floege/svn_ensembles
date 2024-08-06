
import torch
import scipy

from laplace import KronLaplace
from laplace.curvature import AsdlGGN, AsdlEF, GGNInterface
from laplace import Laplace

from stein_classes.loss import calc_loss
from stein_classes.stein_utils import hessian_matvec, kfac_hessian_matvec_block, diag_hessian_matvec_block


from SVN.hessian_approximation import HessianApproximation

class KronHessian(HessianApproximation):
    def compute_hessians(self):
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
            
        return hessian_list
    
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
            H_op = scipy.sparse.linalg.LinearOperator((N*D, N*D), matvec=lambda x: hessian_matvec(x, K_XX, kron_hessians, H2, self.n_parameters, self.device))
            v_svgd_numpy = v_svgd.detach().cpu().flatten().numpy()
            alphas, _ = scipy.sparse.linalg.cg(H_op, v_svgd_numpy, maxiter=self.cg_maxiter)
            alphas = torch.tensor(alphas, dtype=torch.float32).reshape(self.n_particles, self.n_parameters).to(self.device)                
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
       
        return v_svn


    
    def solve_block_linear_system(self,kron_hessians, K_XX, grad_K, v_svgd):

        N = self.n_particles
        D = self.n_parameters

        if self.cfg.SVN.ll:
            last_layer_params = self.calculate_last_layer_params(self, kron_hessians)
       
        alpha_list = []
        for i in range(self.n_particles):
            v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
            squared_kernel = K_XX**2
            if self.cfg.SVN.ll:
                grad_K_i_last_layer = grad_K[i][:, -last_layer_params:]
                H_op_part = scipy.sparse.linalg.LinearOperator((last_layer_params, last_layer_params), matvec=lambda x: kfac_hessian_matvec_block(x, squared_kernel[i][i],grad_K_i_last_layer, kron_list[i], device))
                alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part[-last_layer_params:], maxiter=cg_maxiter)
            else:
                H_op_part = scipy.sparse.linalg.LinearOperator((D, D), matvec=lambda x: kfac_hessian_matvec_block(x, squared_kernel[i][i],grad_K[i], kron_list[i], device))
                alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=self.cg_maxiter)
            alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(self.device)
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
        if self.cfg.SVN.ll:
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, last_layer_params)
            ll_v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, last_layer_params)
            v_svn = v_svgd
            v_svn[:, -last_layer_params:] = ll_v_svn

        else:
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        return v_svn
    
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