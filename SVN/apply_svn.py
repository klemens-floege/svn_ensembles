import torch
import torch.autograd as autograd
import torch.utils.data as data_utils

from stein_classes.loss import calc_loss



def apply_SVN(modellist, parameters, batch, train_dataloader, kernel, device, cfg, optimizer, step, svn_calculator):

    inputs = batch[0].to(device)
    targets = batch[1].to(device)
    n_particles = len(modellist)
    n_parameters = sum(p.numel() for p in modellist[0].parameters() if p.requires_grad)
    param_tensors = [p.view(-1) for p in parameters]  # Flatten

    loss, log_prob = calc_loss(modellist, batch, train_dataloader, cfg, device)
    score_func = autograd.grad(log_prob.sum(), parameters)
    score_tensors = [t.view(-1) for t in score_func]  # Flatten
    score_func_tensor = torch.cat(score_tensors).view(n_particles, n_parameters)  # (n_particles, n_parameters)

    #Gradient Clipping
    if cfg.experiment.grad_clip: 
        norm = score_func_tensor.norm(p=2, dim=1, keepdim=True)  # Compute the norm for each particle
        max_norm = cfg.experiment.grad_clip_max_norm
        norm_clipping_mask = (norm > max_norm)
        norm_clipping_mask = norm_clipping_mask.squeeze()

        for i in range(norm_clipping_mask.shape[0]):
            if bool(norm_clipping_mask[i]):
                score_func_tensor[i] = (score_func_tensor[i] 
                                                / norm[i]) * max_norm

    if cfg.task.dataset in ['mnist', 'fashionmnist'] : 
        inputs_squeezed = inputs #do not squeeue e.g. [Bsz, 1, 28, 28]
    else:
        inputs_squeezed = inputs.squeeze(1)

    if cfg.SVN.classification_likelihood and cfg.task.task_type == 'classification':
        targets = targets.squeeze(1)
        if inputs_squeezed.dtype != torch.float:
            inputs_squeezed = inputs_squeezed.float()
        if targets.dtype != torch.long:
            targets = targets.long()

    hessian_particle_loader = data_utils.DataLoader(
        data_utils.TensorDataset(inputs_squeezed, targets),
        batch_size=cfg.SVN.hessian_calc_batch_size
    )

    svn_calculator.hessian_particle_loader = hessian_particle_loader
    svn_calculator.step = step


    
    hessians = svn_calculator.compute_hessians()

    #Compute the Kernel and its derivative
    K_XX, grad_K = svn_calculator.compute_kernel(hessians, param_tensors)


    #calculate SVGD update:
    v_svgd = -1 * torch.einsum('mn, mo -> no', K_XX, score_func_tensor) / n_particles + torch.mean(grad_K, dim=0) #(n_particles, n_parameters)

    #calculate v_svn update:
    if cfg.SVN.block_diag_approx:
        v_svn = svn_calculator.solve_block_linear_system(hessians, K_XX, grad_K, v_svgd)
    else: 
        v_svn = svn_calculator.solve_linear_system(hessians, K_XX, grad_K, v_svgd)

    #Assign SVN update as torch gradients
    svn_calculator.assign_grad(v_svn)

    return loss
