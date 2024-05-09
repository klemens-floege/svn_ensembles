import os
import torch
import datetime

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, x, y):
        # Convert the numpy arrays to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # Return the size of the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve an item by index
        return self.x[idx], self.y[idx]

# Function to generate base save path
#TODO: think about proper splitting 
def generate_model_path(cfg):
    #date_str = datetime.datetime.now().strftime("%Y-%m-%d")  # Current date to differentiate experiment versions
    date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Path includes the date, dataset, method, and key experiment parameters
    if cfg.experiment.method == 'SVN':
        base_save_path = f"{cfg.task.dataset}/{cfg.experiment.method}_{cfg.SVN.hessian_calc}/_batch{cfg.experiment.batch_size}_ep{cfg.experiment.num_epochs}_lr{cfg.experiment.lr}_block{cfg.SVN.block_diag_approx}_{cfg.SVN.use_curvature_kernel}/{date_time_str}"
    else:
        base_save_path = f"{cfg.task.dataset}/{cfg.experiment.method}/batch{cfg.experiment.batch_size}_ep{cfg.experiment.num_epochs}_lr{cfg.experiment.lr}/{date_time_str}"

    return base_save_path