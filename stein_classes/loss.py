import torch
import torch.nn.functional as F

#train_Dataloader is uselesss
def calc_loss(modellist, batch,
              train_dataloader, cfg, device):

    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)

    dim_problem = cfg.task.dim_problem
    batch_size = inputs.shape[0]
    
    pred_list = []

    if cfg.task.dataset in ['cifar10']:
        inputs = inputs.squeeze(1) #result [Bsz, 3, 32, 32]

    

    for model in modellist:
        logits = model.forward(inputs)
        if cfg.task.task_type == 'regression':
            pred_list.append(logits)
        elif cfg.task.task_type == 'classification':
            #if cfg.task.dim_problem == 1:
            #    probabilities = F.sigmoid(logits)  # Binary classification
            #else:
            #    probabilities = F.softmax(logits, dim=-1)  # Multi-class classification
            #pred_list.append(probabilities)
            pred_list.append(logits)

    pred = torch.cat(pred_list, dim=0)
    pred_reshaped = pred.view(n_particles, batch_size, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]


    #TODO: fix the dimensions on this
    if targets.dim() == 3 and targets.size(1) == 1 and targets.size(2) == 1:
        targets = targets.squeeze(1)
    elif targets.dim() == 2 and targets.size(1) == 1:
        pass  # No need to squeeze
    #else:
    #    raise ValueError("Unexpected shape of 'targets'. It should be either [batch_size, 1, 1] or [batch_size, 1].")

    #targets = targets.squeeze(1).float()

    # Ensure resulting shape is [batch_size, 1]
    #assert targets.shape[1] == 1 and targets.shape[0] == batch_size
    

    targets_expanded = targets.expand(n_particles, batch_size, dim_problem)

    #print('norm', targets)
    #print('expanded', targets_expanded)
    #print('pred exp', pred_reshaped)
    #print(targets_expanded - pred_reshaped)

    

    if cfg.task.task_type == 'regression':

        #print("targets expan: ", targets_expanded.shape)
        #print("pred expan: ", pred_reshaped.shape)
        
        a = True
        if a:
            loss = 0.5 * torch.mean((targets_expanded - pred_reshaped) ** 2, dim=1)
        else: 
            ensemble_pred = torch.mean(pred_reshaped, dim=0)
            loss = 0.5 * torch.mean((targets - ensemble_pred) ** 2, dim=1)


    elif cfg.task.task_type == 'classification':
        
        #loss = torch.stack([F.nll_loss(p, targets.argmax(1)) for p in pred_reshaped])
        #loss = torch.stack([F.nll_loss(F.log_softmax(p), T.argmax(1)) for p in pred[1]])

        targets = targets.squeeze(1)
        loss = torch.stack([F.cross_entropy(pred_reshaped[i], targets.long()) for i in range(pred_reshaped.size(0))])
        
        

    pred_dist_std = cfg.SVN.red_dist_std
    ll = -loss* batch_size / pred_dist_std ** 2
    log_prob = ll 
        
    #print('targets', targets)
    #print('prediciton ', pred_reshaped)
        
    #print('loss', loss)

    
    

    return loss, log_prob