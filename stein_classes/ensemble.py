import torch
import torch.autograd as autograd

from stein_classes.stein_utils import calc_loss

def apply_Ensemble(modellist, parameters, 
              batch, train_dataloader, kernel, device, cfg):
        
    loss, log_prob = calc_loss(modellist, batch, train_dataloader, cfg, device)

    loss.sum().backward()
    return loss