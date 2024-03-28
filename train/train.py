import torch
from tqdm import tqdm
from torch.optim import AdamW

from stein_classes.svn import apply_SVN
from stein_classes.svgd import apply_SVGD
from stein_classes.ensemble import apply_Ensemble

from utils.kernel import RBF



def train(modellist, lr, num_epochs, train_dataloader, eval_dataloader, device, cfg):
  
  
  n_particles = len(modellist)
  K = RBF(cfg.experiment.kernel_width)

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

            dim_problem = targets.shape[1]
            
            pred_list = []
            for i in range(n_particles):
                pred_list.append(modellist[i].forward(inputs))
            pred = torch.cat(pred_list, dim=0)
            pred_reshaped = pred.view(n_particles, -1, dim_problem)
            ensemble_pred = torch.mean(pred_reshaped, dim=0) 
                        
            loss = (ensemble_pred.expand_as(targets)-targets)**2

            
            total_mse = loss.sum() 
            total_samples += inputs.size(0)

    overall_mse = total_mse / total_samples
    #overall_rmse = torch.sqrt(torch.tensor(overall_mse))
    overall_rmse = torch.sqrt(overall_mse.clone().detach())


    print(f"Test Epoch {epoch}: MSE: {overall_mse:.4f}, RMSE: {overall_rmse:.4f}")