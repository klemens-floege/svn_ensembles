import torch
import os
import copy
import time
import wandb
import tabulate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps

from tqdm import tqdm
from torch.optim import AdamW

from stein_classes.svn import apply_SVN
from stein_classes.svgd import apply_SVGD
from stein_classes.ensemble import apply_Ensemble
from stein_classes.wgd import apply_WGD

from utils.kernel import RBF
from utils.eval import regression_evaluate_modellist, classification_evaluate_modellist
from utils.method_utils import create_ann



def train(modellist, lr, num_epochs, train_dataloader, eval_dataloader, device, cfg):

  for model in modellist:
    model.to(device)
    model.train()
  
  columns = ["epc", "eval_accuracy", "eval_cross_entropy", "eval_entropy", "eval_NLL", "eval_ECE", "eval_Brier", "time"]

  
  n_particles = len(modellist)
  K = RBF(cfg.experiment.kernel_width)

  parameters = [p for model in modellist for p in model.parameters() if p.requires_grad ]
  
  n_parameters_per_model = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
  print('number of parameters per model', n_parameters_per_model)
  

  


  #print(parameters)

  #print(type(W)
  optimizer = AdamW(params=parameters, lr=lr)

  #for param_group in optimizer.param_groups:
  #  for param in param_group['params']:
  #      print(param)  # This prints the parameter tensor directly
  #      print(param.data)  # This prints the data of the parameter tensor

  #Early Stopping and loading best eval loss model
  best_mse = float('inf')
  best_epoch = -1  # To track the epoch number of the best MSE
  epochs_since_improvement = 0
  best_modellist_state = None

  
  avg_time = 0
  logged_values = []
  
  global_step = 0  # Initialize a global step counter
  
  print('-------------------------'+'Start training'+'-------------------------')
  for epoch in range(num_epochs):

    optimizer.zero_grad()
    
    start_time = time.time()
    hessians_list = []
    hessiancurv_list = []
    print('='*100)
    print(f'Epoch {epoch}')
    print(f'total num of batches',len(train_dataloader))
    for step, batch in enumerate(tqdm(train_dataloader)):

      #TODO: double check this
      for model in modellist:
      #  model.to(device)
        model.train()


      optimizer.zero_grad()    

      

      if cfg.experiment.method == "SVN":
        loss, hessian_particles, hesscurv = apply_SVN(modellist, parameters, batch, train_dataloader, K, device, cfg)
        
        hessians_list.append(hessian_particles) # hessian_particles = (n_particles,n_params,n_params)
        hessiancurv_list.append(hesscurv) # hesscurv = (n_params,n_params)
        # print("H.shape",H.shape)
      elif cfg.experiment.method == "SVGD":
        loss = apply_SVGD(modellist, parameters, batch, train_dataloader, K, device, cfg)
      elif cfg.experiment.method == "Ensemble":
        loss = apply_Ensemble(modellist, parameters, batch, train_dataloader, K, device, cfg, optimizer)
      elif cfg.experiment.method == "WGD":
         # annealing scheduler only tested for WGD
        ann_sch = create_ann(cfg)
        loss = apply_WGD(modellist, parameters, batch, train_dataloader, K, device, cfg, epoch, ann_sch)
      else:
          print('Approximate Bayesian Inference method not implemented ')
          ValueError("Approximate Bayesian Inference method not implemented ")
          return
            
      wandb.log({"loss": loss.mean().item()})  # Log loss at each batch step
      #global_step += 1  # Increment global step after logging

      optimizer.step()
  
    best_metric_tracker = None
    time_diff = time.time() - start_time
    
    if cfg.task.task_type == 'regression':
      eval_MSE, eval_rmse, eval_NLL = regression_evaluate_modellist(modellist, dataloader=eval_dataloader, device=device, config=cfg)
      best_metric_tracker = eval_NLL
      print(f"Epoch {epoch}: MSE: {eval_MSE:.4f}, RMSE: {eval_rmse:.4f}, NLL: {eval_NLL:.4f}")
    elif cfg.task.task_type == 'classification':
      eval_accuracy, eval_cross_entropy, eval_entropy, eval_NLL, eval_ECE, eval_Brier, eval_AUROC = classification_evaluate_modellist(modellist, dataloader=eval_dataloader, device=device, config=cfg)
      best_metric_tracker = eval_cross_entropy
      print(f"Epoch {epoch}: Acc: {eval_accuracy:.4f}, CrossEntr: {eval_cross_entropy:.4f}, Enrtr: {eval_entropy:.4f}, NLL: {eval_NLL:.4f}, ECE: {eval_ECE:.4f}, Brier: {eval_Brier:.4f}, AUROC: {eval_AUROC:.4f}")

      # Log evaluation metrics after each epoch
      metrics_to_log = {
            "epoch": epoch, "eval_accuracy": eval_accuracy, "eval_cross_entropy": eval_cross_entropy,
            "eval_entropy": eval_entropy, "eval_NLL": eval_NLL, "eval_ECE": eval_ECE, "eval_Brier": eval_Brier,
            "eval_AUROC": eval_AUROC, "time_per_epoch": time.time() - start_time
      }
      wandb.log(metrics_to_log)
      global_step += 1  # Increment to differentiate from batch logging
      

    # Check for improvement
    if best_metric_tracker < best_mse:
        best_mse = best_metric_tracker
        epochs_since_improvement = 0
        best_epoch = epoch  # To track the epoch number of the best MSE
        best_modellist_state = copy.deepcopy(modellist)  # Deep copy to save the state
        if cfg.task.task_type == 'regression':
          print("New best MSE, model saved.")
        elif cfg.task.task_type == 'classification':
          print("New best CrossEntropy, model saved.")
    else:
        epochs_since_improvement += 1
        if cfg.task.task_type == 'regression':
          print(f"No improvement in MSE for {epochs_since_improvement} epochs.")
        elif cfg.task.task_type == 'classification':
          print(f"No improvement in CrossEntropy for {epochs_since_improvement} epochs.")



    # Early stopping
    if epochs_since_improvement >= cfg.experiment.early_stop_epochs and cfg.experiment.early_stopping:
        print("Early stopping triggered.")
        break
    
    
    # saving state of the hessians
    if cfg.experiment.method == "SVN" and cfg.SVN.hessian_calc == "Full" and (epoch in [0,19,39,59,79,99]):
      directory=f'hessian_plots/epoch_{epoch}'
      if not os.path.exists(directory):
          # Create the directory
          os.makedirs(directory)
          print("Directory created successfully!")
      else:
          print("Directory already exists!")

          
      particle_hessians_list_t_start = [hessians_list[0][i,:,:] for i in range(cfg.experiment.n_particles)]
      particle_hessians_list_t_end = [hessians_list[-1][i,:,:] for i in range(cfg.experiment.n_particles)]

      for p in range(len(particle_hessians_list_t_start)):
        plt.matshow(particle_hessians_list_t_start[p],cmap=cmaps.Blues)
        plt.savefig(f'hessian_plots/epoch_{epoch}/particle_{p}_start.pdf')
        plt.close()


      for p in range(len(particle_hessians_list_t_end)):
        plt.matshow(particle_hessians_list_t_end[p],cmap=cmaps.Blues)
        plt.savefig(f'hessian_plots/epoch_{epoch}/particle_{p}_end.pdf')
        plt.close()
      
      plt.matshow(hessiancurv_list[0],cmap=cmaps.Blues)
      plt.savefig(f'hessian_plots/epoch_{epoch}/curvature_start.pdf')
      plt.close()
      plt.matshow(hessiancurv_list[-1],cmap=cmaps.Blues)
      plt.savefig(f'hessian_plots/epoch_{epoch}/curvature_end.pdf')
      plt.close()
    
    elif cfg.experiment.method == "SVN" and cfg.SVN.hessian_calc == "Diag":
      print('not implemented diag saving')
    elif cfg.experiment.method == "SVN" and cfg.SVN.hessian_calc == "Kron":
      print('not implemented kron saving')

    
    avg_time += time_diff


  avg_time = avg_time / num_epochs

  
  



  # At the end of training, or if early stopping was triggered, load the best model list
  modellist = best_modellist_state
  print("Loaded model list based on eval MSE from epoch: ", best_epoch)

  return avg_time