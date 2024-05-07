import torch
import copy
import time

from tqdm import tqdm
from torch.optim import AdamW

from stein_classes.svn import apply_SVN
from stein_classes.svgd import apply_SVGD
from stein_classes.ensemble import apply_Ensemble

from utils.kernel import RBF
from utils.eval import regression_evaluate_modellist, classification_evaluate_modellist



def train(modellist, lr, num_epochs, train_dataloader, eval_dataloader, device, cfg):

  for model in modellist:
    model.to(device)
    model.train()
  
  
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
  

  print('-------------------------'+'Start training'+'-------------------------')
  for epoch in range(num_epochs):

    optimizer.zero_grad()
    
    start_time = time.time()

    print('='*100)
    print(f'Epoch {epoch}')
    for step, batch in enumerate(tqdm(train_dataloader)):

      #TODO: double check this
      for model in modellist:
      #  model.to(device)
        model.train()


      optimizer.zero_grad()    

      

      if cfg.experiment.method == "SVN":
        loss = apply_SVN(modellist, parameters, batch, train_dataloader, K, device, cfg)
      elif cfg.experiment.method == "SVGD":
        loss = apply_SVGD(modellist, parameters, batch, train_dataloader, K, device, cfg)
      elif cfg.experiment.method == "Ensemble":
        loss = apply_Ensemble(modellist, parameters, batch, train_dataloader, K, device, cfg, optimizer)
      else:
          print('Approximate Bayesian Inference method not implemented ')
          ValueError("Approximate Bayesian Inference method not implemented ")
          return
            

      optimizer.step()
  
    best_metric_tracker = None
    
    if cfg.task.task_type == 'regression':
      eval_MSE, eval_rmse, eval_NLL = regression_evaluate_modellist(modellist, dataloader=eval_dataloader, device=device, config=cfg)
      best_metric_tracker = eval_MSE
      print(f"Epoch {epoch}: MSE: {eval_MSE:.4f}, RMSE: {eval_rmse:.4f}, NLL: {eval_NLL:.4f}")
    elif cfg.task.task_type == 'classification':
      eval_accuracy, eval_cross_entropy, eval_entropy, eval_NLL, eval_ECE, eval_Brier = classification_evaluate_modellist(modellist, dataloader=eval_dataloader, device=device, config=cfg)
      best_metric_tracker = eval_cross_entropy
      print(f"Epoch {epoch}: Acc: {eval_accuracy:.4f}, CrossEntr: {eval_cross_entropy:.4f}, Enrtr: {eval_entropy:.4f}, NLL: {eval_NLL:.4f}, ECE: {eval_ECE:.4f}, Brier: {eval_Brier:.4f}")

    

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
    

    time_diff = time.time() - start_time

    avg_time += time_diff

  avg_time = avg_time / num_epochs
  
  # At the end of training, or if early stopping was triggered, load the best model list
  modellist = best_modellist_state
  print("Loaded model list based on eval MSE from epoch: ", best_epoch)

  return avg_time