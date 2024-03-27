import torch
from tqdm import tqdm
from torch.optim import AdamW
import torch.autograd as autograd
from torch.nn.functional import mse_loss
import scipy

from laplace import FullLaplace, KronLaplace, DiagLaplace
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data_utils

from utils.kernel import RBF
from utils.distribution import Unorm_post
from stein_classes.svgd import SVGD
from stein_classes.svn import SVN


def train_modellist(modellist, lr, num_epochs, train_dataloader, eval_dataloader, device, use_SVN=True):
  
  
  n_particles = len(modellist)
  K = RBF()

  parameters = [p for model in modellist for p in model.parameters()]
  
  n_parameters_per_model = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
  print('number of parameters per model', n_parameters_per_model)


  #print(type(W)
  optimizer = AdamW(params=parameters, lr=lr)
  

  print('-------------------------'+'Start training'+'-------------------------')
  for epoch in range(num_epochs):

    optimizer.zero_grad()
    

    print('='*100)
    print(f'Epoch {epoch}')
    for step, batch in enumerate(tqdm(train_dataloader)):

        #ensemble.zero_grad()
        optimizer.zero_grad()        

        X = batch[0]
        T = batch[1]

        dim_problem = T.shape[1]
        

        pred_list = []

        for i in range(n_particles):
           pred_list.append(modellist[i].forward(X))
    
        pred = torch.cat(pred_list, dim=0)

        pred_reshaped = pred.view(n_particles, T.shape[0], dim_problem)
        T_expanded = T.expand(n_particles, T.shape[0], dim_problem)


        loss = 0.5 * torch.mean((T_expanded - pred_reshaped) ** 2, dim=1)


        pred_dist_std = 0.1
        
        ll = -loss*len(train_dataloader) / pred_dist_std ** 2

        log_prob = ll

        score_func = autograd.grad(log_prob.sum(), parameters)
        score_tensors = [t.view(-1) for t in score_func]  # Flatten
        score_func_tensor = torch.cat(score_tensors).view(n_particles, -1)  # (n_particles, n_paramters_per_model)

        
        particle_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X, T), 
            batch_size=2
            )
        
        
        hessians_list = []

        for i in range(n_particles):
            laplace_particle_model = FullLaplace(modellist[i], likelihood='regression')
            laplace_particle_model.fit(particle_loader)
            Hessian = laplace_particle_model.posterior_precision
            hessians_list.append(Hessian)

        hessians_tensor = torch.cat(hessians_list, dim=0)
        hessians_tensor = hessians_tensor.reshape(n_particles, n_parameters_per_model, n_parameters_per_model) #(n_particles, n_parameters_per_model, n_parameters_per_model)


        
        #compute curvature matrix
        #M = torch.mean(hessians_tensor, axis=0)
        M=None

    
        
        param_tensors = [p.view(-1) for p in parameters]  # Flatten
        ensemble_parameters_tensor = torch.cat(param_tensors).view(n_particles, -1)  # Concatenate and reshape

        

        K_XX = K.forward(ensemble_parameters_tensor, ensemble_parameters_tensor.detach(), M=M) #(n_particles, n_particles)
        grad_K = -autograd.grad(K_XX.sum(), ensemble_parameters_tensor)[0] #(n_particles, n_parameters_per_model)

        #print('KXX: ', K_XX.shape)
        #print('grad K: ', grad_K.shape

        
        #v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt) / self.nParticles + np.mean(gkx, axis=0)
        v_svgd = (K_XX.detach().matmul(score_func_tensor) + grad_K) / ensemble_parameters_tensor.size(0) #(n_particles, n_parameters_per_model)

        #print(v_svgd)
        #print(v_svgd.shape)

        
        #H = self._getSteinHessianPosdef(GN_Hmlpt, kx, gkx1)
        H1 = torch.einsum("xy, xz, xbd -> yzbd", K_XX, K_XX, hessians_tensor) #(n_particles, n_particles, n_parametes_per_model, n_parametes_per_model)
        H2 = torch.einsum('zi,zj -> ij', grad_K, grad_K) #(n_parametes_per_model, n_parametes_per_model))
        H1[range(n_particles), range(n_particles)] += H2

        N, _, D, _ = H1.shape  # Extract N and D from the shape of H
        H_reshaped = H1.permute(0, 2, 1, 3).reshape(N*D, N*D)  # Reshape to [N*D, N*D] ,  #(n_particles*n_parametes_per_model, n_particles*n_parametes_per_model)
        H =  H_reshaped / n_particles

        solve_method = 'CG'
        # solve_method = 'Cholesky'

        if solve_method == 'CG':
            v_svgd_flat = v_svgd.flatten().view(-1, 1)  # Ensure it's a column vector
            
            # Directly solve Ax = b using torch.linalg.solve if A is not too large and is positive definite
            alphas = torch.linalg.solve(H, v_svgd_flat)

            #alternative implementation<
            #cg_maxiter = 50
            #alphas = scipy.sparse.linalg.cg(hessians_tensor, v_svgd.flatten(), maxiter=cg_maxiter)[0]
                    
            alphas_reshaped = alphas.view(n_particles, -1) #(n_particles, n_parameters_per_model)
            v_svn = torch.einsum('xd, xn -> nd', alphas_reshaped, K_XX) #(n_particles, n_parameters_per_model)


        elif solve_method == 'Cholesky':
            lamb = 0.01
            H = H1 + NK * lamb
            UH = scipy.linalg.cholesky(H)
            v_svn = self._getSVN_direction(kx, v_svgd, UH)
        
        #update = v_svn * eps
            
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

        
    
        optimizer.step()
        
        if step == 0:          
          loss_str = ', '.join([f'Loss {i} = {loss[i].item():.4f}' for i in range(loss.shape[0])])
          print(f'Train Epoch {epoch}, {loss_str}')
    
    # Evaluation loop
    for i in range(n_particles):
       modellist[i].eval()
    
    total_mse = 0.0
    total_samples = 0

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            inputs, targets = batch
            
            pred_list = []
            for i in range(n_particles):
                pred_list.append(modellist[i].forward(X))
            pred = torch.cat(pred_list, dim=0)
            pred_reshaped = pred.view(n_particles, -1, dim_problem)
            ensemble_pred = torch.mean(pred_reshaped, dim=0) 
                        
            loss = (ensemble_pred.expand_as(targets)-targets)**2

            
            total_mse = loss.sum() 
            total_samples += inputs.size(0)

    overall_mse = total_mse / total_samples
    overall_rmse = torch.sqrt(torch.tensor(overall_mse))

    print(f"Test Epoch {epoch}: MSE: {overall_mse:.4f}, RMSE: {overall_rmse:.4f}")