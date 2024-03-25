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
  n_parameters_per_model = len([p for p in modellist[0].parameters()])
  print(n_parameters_per_model)
  print(len(parameters))

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
        
        print('length dataloader', len(particle_loader))
        
        
        hessians_list = []

        for i in range(n_particles):
            laplace_particle_model = FullLaplace(modellist[i], likelihood='regression')
            laplace_particle_model.fit(particle_loader)
            

            Hessian = laplace_particle_model.posterior_precision

            hessians_list.append(Hessian)

        hessians_tensor = torch.cat(hessians_list, dim=0)
        hessians_tensor = hessians_tensor.reshape(n_particles, n_parameters_per_model, n_parameters_per_model) #(n_particles, n_parameters_per_model, n_parameters_per_model)

        print('hessians_tensor shape: ', hessians_tensor.shape)

        
        

        #compute curvature matrix
        M = torch.mean(hessians_tensor, axis=0)

        gmlpt = score_func
        GN_Hmlpt = hessians_tensor

        
        param_tensors = [p.view(-1) for p in parameters]  # Flatten
        ensemble_parameters_tensor = torch.cat(param_tensors).view(n_particles, -1)  # Concatenate and reshape

        

        K_XX = K.forward(ensemble_parameters_tensor, ensemble_parameters_tensor.detach(), M=M) #(n_particles, n_particles)
        grad_K = -autograd.grad(K_XX.sum(), ensemble_parameters_tensor)[0] #(n_particles, n_parameters_per_model)

        #print('KXX: ', K_XX.shape)
        print('grad K: ', grad_K.shape)


        

        
        #v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt) / self.nParticles + np.mean(gkx, axis=0)
        v_svgd = (K_XX.detach().matmul(score_func_tensor) + grad_K) / ensemble_parameters_tensor.size(0) #(n_particles, n_parameters_per_model)

        #print(v_svgd)
        #print(v_svgd.shape)

        
        #H = self._getSteinHessianPosdef(GN_Hmlpt, kx, gkx1)

        solve_method = 'CG'
        # solve_method = 'Cholesky'

        if solve_method == 'CG':
            cg_maxiter = 50
            #alphas = scipy.sparse.linalg.cg(hessians_tensor, v_svgd.flatten(), maxiter=cg_maxiter)[0].reshape(self.nParticles, self.DoF)
            
            v_svgd_flat = v_svgd.flatten().view(-1, 1)  # Ensure it's a column vector

            print('v_svg flat shape: ', v_svgd_flat.shape)

            print('hessians_tensor: ', hessians_tensor.shape)

            # Directly solve Ax = b using torch.linalg.solve if A is not too large and is positive definite
            alphas = torch.linalg.solve(hessians_tensor, v_svgd_flat)



            cg_maxiter = 50
            alphas = scipy.sparse.linalg.cg(hessians_tensor, v_svgd.flatten(), maxiter=cg_maxiter)[0]
            
            print('alphas: ', alphas)
            print('alphas shape: ', alphas.shape)

            
            # Reshape alphas if necessary, depending on subsequent usage
            alphas_reshaped = alphas.view(n_Particles, self.DoF)
            
            
            v_svn = contract('xd, xn -> nd', alphas, kx)

        elif solve_method == 'Cholesky':
            lamb = 0.01
            H = H1 + NK * lamb
            UH = scipy.linalg.cholesky(H)
            v_svn = self._getSVN_direction(kx, v_svgd, UH)
        
        update = v_svn * eps
    


        
        

        
        if step == 0:          
          loss_str = ', '.join([f'Loss {i+1} = {losses[i][0]:.4f}' for i in range(len(losses))])
          print(f'Train Epoch {epoch}, {loss_str}')
    
    # Evaluation loop
    for i in range(n_particles):
       modellist[i].eval()
    
    total_mse = 0.0
    total_samples = 0

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            inputs, targets = batch
            outputs = ensemble.forward(inputs)
            
            #average prediction over all particles
            ensemble_pred = torch.mean(outputs[0], dim=0) 
            
            #mse = mse_loss(ensemble_pred, targets)

            #loss = 0.5*torch.mean(F.mse_loss(ensemble_pred, targets, reduction='none'), 1)
            #print(ensemble_pred.shape)
            #print(targets.shape)
            #print((ensemble_pred.expand_as(targets)-targets)**2)
            loss = (ensemble_pred.expand_as(targets)-targets)**2

            #print(a.sum())

            #loss = 0.5*torch.mean((ensemble_pred.expand_as(targets)-targets)**2,1)
            

            
            total_mse = loss.sum() 
            total_samples += inputs.size(0)

    overall_mse = total_mse / total_samples
    overall_rmse = torch.sqrt(torch.tensor(overall_mse))

    print(f"Test Epoch {epoch}: MSE: {overall_mse:.4f}, RMSE: {overall_rmse:.4f}")