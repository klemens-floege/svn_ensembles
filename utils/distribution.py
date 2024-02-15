import torch
import torch.nn.functional as F


class Unorm_post():

    def __init__(self, ensemble, prior, config, n_train,add_prior = False):
        self.prior = prior
        self.ensemble = ensemble
        self.config = config
        self.num_train = n_train
        self.add_prior = add_prior


    def log_prob(self, particles, X, T, return_loss=False, return_pred = False, pred_idx = 1):

        
        pred = self.ensemble.forward(X, particles)

        if self.ensemble.net.classification:
            if pred_idx == 1:

                loss = torch.stack([F.nll_loss(F.log_softmax(p), T.argmax(1)) for p in pred[1]])
            else:

                loss = (-(T.expand_as(pred[1])*torch.log(pred[0]+1e-15))).max(2)[0].sum(1)/X.shape[0]

            #pred = F.softmax(pred[1],2) #I have to do this to allow derivative and to not have nans 
        else:
            #loss = 0.5*torch.mean(F.mse_loss(pred[0], T, reduction='none'), 1)

            loss = 0.5*torch.mean((T.expand_as(pred[0])-pred[0])**2,1)


            #print('loss shape', loss.shape)
            
            #print(pred[0].shape)
            #print(T.shape)
            #print(F.mse_loss(pred[0], T, reduction='none'))
            #loss = 0.5*torch.mean((T.expand_as(pred[0])-pred[0])**2,1)
            #ensemble_losses = []

            # Calculate MSE for each ensemble member
            #for i in range(pred[0].size(0)):  # Loop over ensemble members
            #    # Calculate MSE
            #    mse_loss = 0.5 * torch.mean((T - pred[0][i])**2)
            #    ensemble_losses.append(mse_loss)

            #ensemble_losses_tensor = torch.tensor(ensemble_losses)
            #loss = ensemble_losses_tensor

        pred_dist_std = 0.1
        #ll = -loss*self.num_train / self.config.pred_dist_std ** 2
        #print('loss', loss)
        ll = -loss*self.num_train / pred_dist_std ** 2
        #print('ll', ll)
        #ll = loss

        if particles is None:
            particles = self.ensemble.particles

        if self.add_prior:
            log_prob = torch.add(self.prior.log_prob(particles).sum(1), ll)
        else:
            log_prob = ll

        
        if return_loss:
            return torch.mean(loss),pred[0]
        elif return_pred:
            return log_prob,pred #0 softmax, 1 is logit
        else:
            return log_prob