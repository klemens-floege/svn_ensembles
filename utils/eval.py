import torch
import numpy as np
from tqdm import tqdm  

# Evaluation loop
def evaluate_modellist(modellist, dataloader):
    
    n_particles = len(modellist)
    for model in modellist:
        model.eval()
    
    total_mse = 0.0
    total_nll = 0.0  # Initialize total NLL
    total_samples = 0

    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs, targets = batch

            dim_problem = targets.shape[1]
            
            pred_list = []
            for i in range(n_particles):
                pred_list.append(modellist[i].forward(inputs))
            pred = torch.cat(pred_list, dim=0)
            pred_reshaped = pred.view(n_particles, -1, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]

            # Mean prediction
            ensemble_pred = torch.mean(pred_reshaped, dim=0) 

             # Variance as a proxy for uncertainty
            ensemble_variance = pred_reshaped.var(dim=0) + 1e-6  # Adding a small constant for numerical stability
            

             # MSE
            mse_loss = (ensemble_pred - targets) ** 2
            
            # NLL assuming Gaussian distribution
            nll_loss = 0.5 * torch.log(2 * np.pi * ensemble_variance) + (mse_loss / (2 * ensemble_variance))
            total_nll += nll_loss.sum().item()

            total_mse = mse_loss.sum() 
            total_samples += inputs.size(0)

    
    eval_MSE = total_mse / total_samples
    #eval_rmse = torch.sqrt(eval_MSE.clone().detach())
    eval_RMSE = np.sqrt(eval_MSE)
    eval_NLL = total_nll / total_samples  # Average NLL per data point

    return eval_MSE, eval_RMSE, eval_NLL
    