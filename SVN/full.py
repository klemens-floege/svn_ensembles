import torch
import scipy

from laplace import FullLaplace
from laplace.curvature import AsdlGGN, AsdlEF, GGNInterface



from SVN.hessian_approximation import HessianApproximation

class FullHessian(HessianApproximation):
    def compute_hessians(self):
        hessian_list = []
        for i in range(self.n_particles):

            if self.cfg.SVN.classification_likelihood:
                laplace_particle_model = FullLaplace(self.modellist[i], 
                                                        likelihood=self.cfg.task.task_type)
            else:
                laplace_particle_model = FullLaplace(self.modellist[i], 
                                                        likelihood='regression')
            laplace_particle_model.fit(self.hessian_particle_loader)
            particle_hessian = laplace_particle_model.posterior_precision
            hessian_list.append(particle_hessian)
            
        hessians_tensor = torch.cat(hessian_list, dim=0)
        hessians_tensor = hessians_tensor.reshape(self.n_particles, self.n_parameters, self.n_parameters) #(n_particles, n_parameters, n_parameters)
        return hessians_tensor
    
    def compute_kernel(self, hessians_tensor, parameter_tensors):

        if self.cfg.SVN.use_curvature_kernel == "use_curvature":
            M = torch.mean(hessians_tensor, axis=0)
        else:
            M=None

        ensemble_parameters_tensor = torch.cat(parameter_tensors).view(self.n_particles, -1)  # Concatenate and reshape

        h = self.cfg.experiment.kernel_width        
        displacement_tensor = ensemble_parameters_tensor[:, None, :] - ensemble_parameters_tensor[None, :, :]
        
        if M is not None:
            epsilon = 1e-6  # A small value; adjust as necessary
            M_reg = M + epsilon * torch.eye(M.size(0), device=M.device)
            U = torch.linalg.cholesky(M_reg)
            #U = torch.linalg.cholesky(M)
            ensemble_parameters_tensor = torch.matmul(U, ensemble_parameters_tensor.T).T
            displacement_tensor = torch.matmul(displacement_tensor, M)
            
        squared_distances = torch.cdist(ensemble_parameters_tensor, ensemble_parameters_tensor, p=2) ** 2
        K_XX = torch.exp(-squared_distances / h) #(n_particles, n_particles)
        grad_K = -2 * (K_XX.unsqueeze(-1) * displacement_tensor) / h  #(n_particles, n_particles, n_parameters)

        return K_XX, grad_K


    def solve_linear_system(self, hessians_tensor, K_XX, grad_K, v_svgd):
        H1 = torch.einsum("xy, xz, xbd -> yzbd", K_XX, K_XX, hessians_tensor) #(n_particles, n_particles, n_parametes_per_model, n_parametes_per_model)
        H2 = torch.einsum('xzi,xzj -> zij', grad_K, grad_K) #(n_particles, n_parametes_per_model, n_parametes_per_model)
        
        #adds H2 values to diagonal of H1
        H1[range(self.n_particles), range(self.n_particles)] += H2

        N, _, D, _ = H1.shape  # Extract N and D from the shape of H
        H_reshaped = H1.permute(0, 2, 1, 3).reshape(N*D, N*D)  # Reshape to [N*D, N*D] ,  #(n_particles*n_parametes_per_model, n_particles*n_parametes_per_model)
        H =  H_reshaped / self.n_particles
        H_op = H.detach().cpu().numpy()
        v_svgd_numpy = v_svgd.detach().cpu().flatten().numpy()
        alphas, _ = scipy.sparse.linalg.cg(H_op, v_svgd_numpy, maxiter=self.cg_maxiter)
        alphas = torch.tensor(alphas, dtype=torch.float32).reshape(self.n_particles, self.n_parameters).to(self.device)        
        if self.cfg.SVN.ll:
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, last_layer_params)
            ll_v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, last_layer_params)
            v_svn = v_svgd
            v_svn[:, -self.last_layer_params:] = ll_v_svn
        else: 
            alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        return v_svn


    
    def solve_block_linear_system(self,hessians_tensor, K_XX, grad_K, v_svgd):
        alpha_list = []
        for i in range(self.n_particles):
            v_svgd_part = v_svgd[i].squeeze().detach().cpu().flatten().numpy()
            squared_kernel = K_XX**2
            H_part= hessians_tensor[i]*  squared_kernel[i][i] + torch.matmul(grad_K[i].T, grad_K[i])
            H_op_part = H_part.detach().cpu().numpy()
            alpha_part, _ = scipy.sparse.linalg.cg(H_op_part, v_svgd_part, maxiter=self.cg_maxiter)
            alpha_part = torch.tensor(alpha_part, dtype=torch.float32).to(self.device)
            alpha_list.append(alpha_part)
        alphas = torch.stack(alpha_list, dim=0).view(self.n_particles, -1)
        alphas_reshaped = alphas.view(self.n_particles, -1) #(n_particles, n_parameters)
        v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters)
        return v_svn