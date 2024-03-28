import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from utils.kernel import RBF
from utils.synth_data import get_sine_data
from utils.plot import plot_modellist
from train.train import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def run_experiment(cfg):


    if cfg.experiment.dataset == "sine":
        x_train, y_train, x_test, y_test = get_sine_data(n_samples=cfg.experiment.n_samples, seed= cfg.experiment.seed)
    else: 
        ValueError("The configured dataset is not yet implemented")
    


    layer_sizes = cfg.experiment.layer_sizes

    print("layer sizes: ", layer_sizes)

    n_particles = cfg.experiment.n_particles

    modellist = []

    for _ in range(n_particles):
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Add ReLU activation, except for the output layer
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers)
        modellist.append(model)

    

    # Split the data into training and evaluation sets
    x_train_split, x_eval_split, y_train_split, y_eval_split = train_test_split(x_train, y_train, test_size=cfg.experiment.train_test_split, random_state=cfg.experiment.seed)

    # Create instances of the SineDataset for each set
    train_dataset = Dataset(x_train_split, y_train_split)
    eval_dataset = Dataset(x_eval_split, y_eval_split)
    
    

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)

    lr = cfg.experiment.lr
    num_epochs = cfg.experiment.num_epochs

    metrics = train(modellist, lr, num_epochs, train_dataloader, eval_dataloader, device, cfg)

    
    plot_save_path = cfg.experiment.plot_save_path
    plot_modellist(modellist, x_train, y_train, x_test, y_test, plot_save_path)
