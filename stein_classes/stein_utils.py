import torch


def calc_loss(modellist, batch,
              train_dataloader, cfg):

    inputs = batch[0]
    targets = batch[1]

    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)

    dim_problem = targets.shape[1]
    

    pred_list = []

    for i in range(n_particles):
        pred_list.append(modellist[i].forward(inputs))

    pred = torch.cat(pred_list, dim=0)

    pred_reshaped = pred.view(n_particles, targets.shape[0], dim_problem)
    T_expanded = targets.expand(n_particles, targets.shape[0], dim_problem)

    #compute MSE loss
    loss = 0.5 * torch.mean((T_expanded - pred_reshaped) ** 2, dim=1)
    pred_dist_std = cfg.SVN.red_dist_std
    ll = -loss*len(train_dataloader) / pred_dist_std ** 2
    log_prob = ll

    return loss, log_prob