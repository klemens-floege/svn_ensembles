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
    else:
        raise ValueError("Unexpected shape of 'targets'. It should be either [batch_size, 1, 1] or [batch_size, 1].")

    # Ensure resulting shape is [batch_size, 1]
    assert targets.shape[1] == 1 and targets.shape[0] == batch_size
    
    targets_expanded = targets.expand(n_particles, targets.shape[0], dim_problem)
    

    if cfg.task.task_type == 'regression':
        
        loss = 0.5 * torch.mean((targets_expanded - pred_reshaped) ** 2, dim=1)

    elif cfg.task.task_type == 'classification':

        #loss = torch.stack([F.nll_loss(p, targets.argmax(1)) for p in pred_reshaped])
        #loss = torch.stack([F.nll_loss(F.log_softmax(p), T.argmax(1)) for p in pred[1]])
        loss = torch.stack([F.cross_entropy(pred_reshaped[i], targets.squeeze(1).long()) for i in range(pred_reshaped.size(0))])
        

    #pred_dist_std = cfg.SVN.red_dist_std
    #ll = -loss*len(train_dataloader) / pred_dist_std ** 2
    #log_prob = ll
    

    return loss#, log_prob