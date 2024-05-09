import torch
import numpy as np
from tqdm import tqdm  

import torch.nn.functional as F
from torch.nn import MSELoss
 
from torchmetrics.classification import MulticlassCalibrationError, AUROC
from utils.brier import brier_scores

# Evaluation loop
def regression_evaluate_modellist(modellist, dataloader, device, config):
    
    n_particles = len(modellist)
    for model in modellist:
        model.eval()
    
    mse_loss = MSELoss()  # Initialize the MSE loss function


    total_mse = 0.0
    total_nll = 0.0  # Initialize total NLL
    total_samples = 0
    n_batches = 0

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
            ensemble_variance = pred_reshaped.var(dim=0) + 1e-6  # Adding a small constant for numerical stability #[batch_size, dim_problem]

            
            loss = mse_loss(ensemble_pred, targets) #'mean' reduction is default

            # NLL assuming Gaussian distribution
            nll_loss = torch.nn.functional.gaussian_nll_loss(ensemble_pred, targets, ensemble_variance) #'mean' reduction is default
            
            
            total_nll += nll_loss.sum().item() * inputs.size(0) 
            total_mse = loss.item() * inputs.size(0) 
            total_samples += inputs.size(0)

            n_batches+= 1

    
    eval_MSE = total_mse / total_samples
    #eval_MSE = total_mse / n_batches
    eval_RMSE = torch.sqrt(torch.tensor(eval_MSE)).item()
    #eval_RMSE = np.sqrt(eval_MSE)
    eval_NLL = total_nll / total_samples  # Average NLL per data point
    #eval_NLL = total_nll / n_batches  # Average NLL per data point

    return eval_MSE, eval_RMSE, eval_NLL
    

def classification_evaluate_modellist(modellist, dataloader, device, config):
    
    n_particles = len(modellist)
    for model in modellist:
        model.eval()
    
    total_samples = 0
    total_loss = 0.0
    total_correct = 0
    total_nll = 0.0  # Initialize total NLL
    total_entropy = 0.0
    total_ece = 0.0
    total_brier = 0.0
    total_auroc = 0.0
    

    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            dim_problem = config.task.dim_problem
            batch_size = inputs.shape[0]

            if config.task.dataset in ['cifar10']:
                inputs = inputs.squeeze(1) #result [Bsz, 3, 32, 32]
            
            logits_list = []
            probabilities_list = []

            for model in modellist:
                
                particle_logits = model.forward(inputs) 
                logits_list.append(particle_logits)
                if config.task.dim_problem == 1:
                    particle_probabilities = F.Sigmoid(particle_logits)  # Binary classification
                else:
                    particle_probabilities = F.softmax(particle_logits, dim=-1)# Multi-class classification
                probabilities_list.append(particle_probabilities)

            logits = torch.cat(logits_list, dim=0)
            probabilities = torch.cat(probabilities_list, dim=0)
            logits_reshaped = logits.view(n_particles, batch_size, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]
            probabilities_reshaped = probabilities.view(n_particles, batch_size, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]


            # Mean prediction
            logits_ensemble_pred = torch.mean(logits_reshaped, dim=0) 
            probs_ensemble_pred = torch.mean(probabilities_reshaped, dim=0) 


            if targets.dim() == 3 and targets.size(1) == 1 and targets.size(2) == 1:
                targets = targets.squeeze(1)
            elif targets.dim() == 2 and targets.size(1) == 1:
                pass  # No need to squeeze
            else:
                raise ValueError("Unexpected shape of 'targets'. It should be either [batch_size, 1, 1] or [batch_size, 1].")

            # Ensure resulting shape is [batch_size, 1]
            assert targets.shape[1] == 1 and targets.shape[0] == batch_size

            # Variance as a proxy for uncertainty
            #ensemble_variance = logits_reshaped.var(dim=0) + 1e-6  # Adding a small constant for numerical stability

            targets = targets.squeeze(1).long()  # Squeeze and convert to Long if necessary
        
            loss = torch.nn.functional.cross_entropy(logits_ensemble_pred, targets)
            total_loss += loss.item()

          
            _, predicted_labels = torch.max(probs_ensemble_pred, 1)
          
            
            entropy = -(probs_ensemble_pred * torch.log(probs_ensemble_pred + 1e-6)).sum(dim=1).mean()
            

            log_pred_reshaped = torch.log(probs_ensemble_pred + 1e-15)
            #nll = torch.stack([torch.nn.functional.nll_loss(p, targets) for p in log_pred_reshaped])
            nll = torch.nn.functional.nll_loss(log_pred_reshaped, targets)

            ECE_loss = MulticlassCalibrationError(num_classes=config.task.dim_problem, n_bins=15, norm='l1')
            ece = ECE_loss(probs_ensemble_pred, targets)
            brier = brier_scores(probs_ensemble_pred, targets, config.task.dim_problem)

            if config.task.dim_problem == 2:
                auroc = AUROC(task="binary")
                auroc(probs_ensemble_pred, targets)
            else: 
                auroc = AUROC(task="multiclass", num_classes=config.task.dim_problem)
                auroc(probs_ensemble_pred, targets)
            
            total_correct += (predicted_labels == targets).sum().item()
            total_nll += nll.sum().item() * inputs.size(0)
            total_ece += ece.sum().item() * inputs.size(0)
            total_brier += brier.sum().item()  * inputs.size(0)
            total_loss += loss.sum() * inputs.size(0)
            total_entropy += entropy.item()
            total_auroc += auroc.sum().item() * inputs.size(0)
            total_samples += inputs.size(0)


    eval_accuracy = total_correct / total_samples    
    eval_cross_entropy = total_loss / total_samples
    eval_entropy = total_entropy / total_samples
    eval_NLL = total_nll / total_samples  # Average NLL per data point
    eval_ECE = total_ece / total_samples
    eval_Brier = total_brier / total_samples
    eval_AUROC = total_auroc / total_samples

    return eval_accuracy, eval_cross_entropy, eval_entropy, eval_NLL, eval_ECE, eval_Brier, eval_AUROC