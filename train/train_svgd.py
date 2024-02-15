
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn.functional import mse_loss

from utils.kernel import RBF
from utils.distribution import Unorm_post
from stein_classes.svgd import SVGD
from stein_classes.svn import SVN


def train(ensemble, lr, num_epochs, train_dataloader, eval_dataloader, device, use_SVN=True):
  W = ensemble.particles
  print("W shape before training: ", W.shape)
  K = RBF()

  #print(type(W)
  optimizer = AdamW(params=[W], lr=lr)
  

  print('-------------------------'+'Start training'+'-------------------------')
  for epoch in range(num_epochs):

    optimizer.zero_grad()
    prior_variance = 0.01 #replace with config.prior_variance
    #prior = torch.distributions.normal.Normal(torch.zeros(len_params).to(device),torch.ones(len_params).to(device) * prior_variance)
    
    prior = torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                              torch.ones(ensemble.net.num_params).to(device) * prior_variance)

    P = Unorm_post(ensemble, prior, None, len(train_dataloader.dataset))

    print('='*100)
    print(f'Epoch {epoch}')
    for step, batch in enumerate(tqdm(train_dataloader)):

        #ensemble.zero_grad()
        optimizer.zero_grad()        

        X = batch[0]
        T = batch[1]

        #TODO: make this varaibel depending on methods
        if use_SVN:
            method = SVN(P,K,optimizer)
        else:
            method = SVGD(P,K,optimizer)

        phi_output, score_func, grad_K, losses = method.step(W, X, T)
        
        

        
        if step == 0:          
          loss_str = ', '.join([f'Loss {i+1} = {losses[i][0]:.4f}' for i in range(len(losses))])
          print(f'Train Epoch {epoch}, {loss_str}')
    
    # Evaluation loop
    ensemble.net.eval()  # Set the model to evaluation mode
    
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