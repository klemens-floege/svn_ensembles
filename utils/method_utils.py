import torch 


def create_ann(config): 
    if config.experiment.ann_sched_type == 'linear': 
        ann_sch = torch.cat([torch.linspace(0,config.experiment.ann_sched_gamma,config.experiment.ann_steps),config.experiment.ann_sched_gamma*torch.ones(config.experiment.num_epochs-config.experiment.ann_steps)]) 
    elif config.experiment.ann_sched_type == 'hyper': 
        ann_sch =torch.cat([torch.tanh((torch.linspace(0,config.experiment.ann_steps,config.experiment.ann_steps)*1.3/config.experiment.ann_steps)**10),config.experiment.ann_sched_gamma*torch.ones(config.experiment.num_epochs-config.experiment.ann_steps)])
    elif config.experiment.ann_sched_type == 'cyclic':
        # ann_sch = torch.cat([torch.tensor([cosine_annealing(a,config.experiment.ann_steps,5,1)**10 for a in range(config.experiment.ann_steps)]),config.experiment.ann_sched_gamma*torch.ones(config.experiment.num_epochs-config.experiment.ann_steps)])
        print("not implemented ",config.ann_sched_type, "using default")
        ann_sch = config.experiment.ann_sched_gamma*torch.ones(config.experiment.num_epochs)
    elif config.experiment.ann_sched_type == 'None':
        ann_sch = config.experiment.ann_sched_gamma*torch.ones(config.experiment.num_epochs)
    return ann_sch

# cosine annealing learning rate schedule
# def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
#     epochs_per_cycle = floor(n_epochs/n_cycles)
#     cos_inner =(epoch % epochs_per_cycle)/ (epochs_per_cycle)
#     return (cos_inner)