import torch
import copy

from tqdm import tqdm
from torch.optim import AdamW

from stein_classes.svn import apply_SVN
from stein_classes.svgd import apply_SVGD
from stein_classes.ensemble import apply_Ensemble

from utils.kernel import RBF
from utils.eval import evaluate_modellist



def train(modellist, lr, num_epochs, train_dataloader, eval_dataloader, device, cfg):
  
  
  n_particles = len(modellist)
  K = RBF(cfg.experiment.kernel_width)

  parameters = [p for model in modellist for p in model.parameters()]
  
  n_parameters_per_model = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
  print('number of parameters per model', n_parameters_per_model)


  #print(type(W)
  optimizer = AdamW(params=parameters, lr=lr)

  #Early Stopping and loading best eval loss model
  best_mse = float('inf')
  best_epoch = -1  # To track the epoch number of the best MSE
  epochs_since_improvement = 0
  best_modellist_state = None
  

  print('-------------------------'+'Start training'+'-------------------------')
  for epoch in range(num_epochs):

    optimizer.zero_grad()
    

    print('='*100)
    print(f'Epoch {epoch}')
    for step, batch in enumerate(tqdm(train_dataloader)):


        optimizer.zero_grad()        

        if cfg.experiment.method == "SVN":
          loss = apply_SVN(modellist, parameters, batch, train_dataloader, K, device, cfg)
        elif cfg.experiment.method == "SVGD":
          loss = apply_SVGD(modellist, parameters, batch, train_dataloader, K, device, cfg)
        elif cfg.experiment.method == "Ensemble":
          loss = apply_Ensemble(modellist, parameters, batch, train_dataloader, K, device, cfg)
        else:
           ValueError("Approximate Bayesian Inference method not implemented ")

            
        optimizer.step()
        
        #if step == 0:          
        #  loss_str = ', '.join([f'Loss {i} = {loss[i].item():.4f}' for i in range(loss.shape[0])])
        #  print(f'Train Epoch {epoch}, {loss_str}')

    

    eval_MSE, eval_rmse, eval_NLL = evaluate_modellist(modellist, dataloader=eval_dataloader)

    print(f"Epoch {epoch}: MSE: {eval_MSE:.4f}, RMSE: {eval_rmse:.4f}, NLL: {eval_NLL:.4f}")

    # Check for improvement
    if eval_MSE < best_mse:
        best_mse = eval_MSE
        epochs_since_improvement = 0
        best_epoch = epoch  # To track the epoch number of the best MSE
        best_modellist_state = copy.deepcopy(modellist)  # Deep copy to save the state
        print("New best MSE, model saved.")
    else:
        epochs_since_improvement += 1
        print(f"No improvement in MSE for {epochs_since_improvement} epochs.")



    # Early stopping
    if epochs_since_improvement >= cfg.experiment.early_stop_epochs and cfg.experiment.early_stopping:
        print("Early stopping triggered.")
        break

  # At the end of training, or if early stopping was triggered, load the best model list
  modellist = best_modellist_state
  print("Loaded model list based on eval MSE from epoch: ", best_epoch)