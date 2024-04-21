import torch
import torch.autograd as autograd

from stein_classes.stein_utils import calc_loss

def apply_Ensemble(modellist, parameters, 
              batch, train_dataloader, kernel, device, cfg, optimizer):
        
    loss, log_prob = calc_loss(modellist, batch, train_dataloader, cfg, device)

    #print('loss', loss)

    loss.sum().backward()

    
    """for param in parameters:
        if param.grad is not None:
            print("Gradient for parameter:", param.grad)
        else:
            print("No gradient for parameter")  """

    #optimizer.step()

    return loss