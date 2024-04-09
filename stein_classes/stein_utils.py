import torch


def calc_loss(modellist, batch,
              train_dataloader, cfg, device):

    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)

    dim_problem = targets.shape[1]
    
    pred_list = []

    for i in range(n_particles):
        pred_list.append(modellist[i].forward(inputs))

    pred = torch.cat(pred_list, dim=0)
    pred_reshaped = pred.view(n_particles, -1, dim_problem) # Stack to get [n_particles, batch_size, dim_problem]

    # Mean prediction
    ensemble_pred = torch.mean(pred_reshaped, dim=0) 

    mse_loss = (ensemble_pred - targets) ** 2
    loss = mse_loss
    pred_dist_std = cfg.SVN.red_dist_std
    ll = -loss*len(train_dataloader) / pred_dist_std ** 2
    log_prob = ll

    return loss, log_prob