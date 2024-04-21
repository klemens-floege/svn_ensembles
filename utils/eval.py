import torch
import numpy as np
from tqdm import tqdm  

# Evaluation loop
def regression_evaluate_modellist(modellist, dataloader, device, config):
    
    n_particles = len(modellist)
    for model in modellist:
        model.eval()
    
    total_mse = 0.0
    total_nll = 0.0  # Initialize total NLL
    total_samples = 0

    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            dim_problem = targets.shape[1]
            batch_size = inputs.shape[0]
            
            pred_list = []
            for i in range(n_particles):
                if config.task.task_type == 'regression': 
                    pred_list.append(modellist[i].forward(inputs))
                elif config.task.task_type == 'classification': 
                    logits = modellist[i](inputs)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                    pred_list.append(probabilities)

            pred = torch.cat(pred_list, dim=0)
            pred_reshaped = pred.view(n_particles, batch_size, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]

            # Mean prediction
            ensemble_pred = torch.mean(pred_reshaped, dim=0) 


            if targets.dim() == 3 and targets.size(1) == 1 and targets.size(2) == 1:
                targets = targets.squeeze(1)
            elif targets.dim() == 2 and targets.size(1) == 1:
                pass  # No need to squeeze
            else:
                raise ValueError("Unexpected shape of 'targets'. It should be either [batch_size, 1, 1] or [batch_size, 1].")

            # Ensure resulting shape is [batch_size, 1]
            assert targets.shape[1] == 1 and targets.shape[0] == batch_size


             # Variance as a proxy for uncertainty
            ensemble_variance = pred_reshaped.var(dim=0) + 1e-6  # Adding a small constant for numerical stability


            mse_loss = (ensemble_pred - targets) ** 2
            loss = mse_loss
            
            # NLL assuming Gaussian distribution
            nll_loss = 0.5 * torch.log(2 * np.pi * ensemble_variance) + (loss / (2 * ensemble_variance))
            total_nll += nll_loss.sum().item()

            total_mse = loss.sum() 
            total_samples += inputs.size(0)

    
    eval_MSE = total_mse / total_samples
    eval_RMSE = torch.sqrt(eval_MSE.clone().detach())
    #eval_RMSE = np.sqrt(eval_MSE)
    eval_NLL = total_nll / total_samples  # Average NLL per data point

    return eval_MSE, eval_RMSE, eval_NLL
    

def classification_evaluate_modellist(modellist, dataloader, device, config):
    
    n_particles = len(modellist)
    for model in modellist:
        model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_nll = 0.0  # Initialize total NLL
    total_entropy = 0.0
    total_samples = 0

    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            dim_problem = config.task.dim_problem
            batch_size = inputs.shape[0]
            
            pred_list = []
            for i in range(n_particles):
                
                logits = modellist[i](inputs)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                pred_list.append(probabilities)

            pred = torch.cat(pred_list, dim=0)
            pred_reshaped = pred.view(n_particles, batch_size, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]

            # Mean prediction
            ensemble_pred = torch.mean(pred_reshaped, dim=0) 


            if targets.dim() == 3 and targets.size(1) == 1 and targets.size(2) == 1:
                targets = targets.squeeze(1)
            elif targets.dim() == 2 and targets.size(1) == 1:
                pass  # No need to squeeze
            else:
                raise ValueError("Unexpected shape of 'targets'. It should be either [batch_size, 1, 1] or [batch_size, 1].")

            # Ensure resulting shape is [batch_size, 1]
            assert targets.shape[1] == 1 and targets.shape[0] == batch_size


             # Variance as a proxy for uncertainty
            ensemble_variance = pred_reshaped.var(dim=0) + 1e-6  # Adding a small constant for numerical stability

            targets = targets.squeeze(1).long()  # Squeeze and convert to Long if necessary


            loss = torch.nn.functional.cross_entropy(ensemble_pred, targets)
            total_loss += loss.item()

            _, predicted_labels = torch.max(ensemble_pred, 1)
            
            entropy = -(ensemble_pred * torch.log(ensemble_pred + 1e-6)).sum(dim=1).mean()

            log_probs = torch.log(ensemble_pred + 1e-6)  # Adding a small constant to prevent log(0)
            nll = -log_probs[range(targets.size(0)), targets].mean()  # Negative log-likelihood

            
            total_correct += (predicted_labels == targets).sum().item()
            total_nll += nll.sum().item()
            total_loss += loss.sum() 
            total_entropy += entropy.item()
            total_samples += inputs.size(0)

    
    eval_cross_entropy = total_loss / total_samples
    eval_entropy = total_entropy / total_samples
    eval_NLL = total_nll / total_samples  # Average NLL per data point

    return eval_cross_entropy, eval_entropy, eval_NLL