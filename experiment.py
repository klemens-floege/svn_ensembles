import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from utils.kernel import RBF
from utils.data import get_sine_data, get_gap_data, load_yacht_data
from utils.plot import plot_modellist
from utils.eval import regression_evaluate_modellist, classification_evaluate_modellist
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


    if cfg.task.dataset == "sine":
        x_train, y_train, x_test, y_test = get_sine_data(n_samples=cfg.experiment.n_samples, seed= cfg.experiment.seed)
    elif cfg.task.dataset == "gap":
        x_train, y_train, x_test, y_test = get_gap_data(n_samples=cfg.experiment.n_samples, seed= cfg.experiment.seed)
    else: 
        ValueError("The configured dataset is not yet implemented")
    
    
    # Example dimensions, replace with x_train.shape[1] and y_train.shape[1] or 1 as appropriate
    x_train_feature_dim = x_train.shape[1]  # Number of features in x_train
    y_train_feature_dim = 1  # Assuming y_train is a vector; if it's a 2D array with one column, this is correct

    # Dynamically adjust layer sizes
    n_particles = cfg.experiment.n_particles
    hidden_layers = cfg.experiment.hidden_layers


    layer_sizes = []
    layer_sizes.append(x_train_feature_dim)  # Set the first layer size to match the feature dimension of x_train
    for k in range(len(hidden_layers)):
        layer_sizes.append(hidden_layers[k])
    layer_sizes.append(y_train_feature_dim)  # Set the last layer size to match the feature dimension of y_train (or 1 if y_train is a vector)

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


    print("layer sizes: ", layer_sizes)

    

    
    # Split the data into training and evaluation sets
    x_train_split, x_eval_split, y_train_split, y_eval_split = train_test_split(x_train, y_train, test_size=0.2, random_state=cfg.experiment.seed)

    # Create instances of the SineDataset for each set
    train_dataset = Dataset(x_train_split, y_train_split)
    eval_dataset = Dataset(x_eval_split, y_eval_split)
    test_dataset = Dataset(x_test, y_test)
    
    

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)

    lr = cfg.experiment.lr
    num_epochs = cfg.experiment.num_epochs

    avg_train_time_per_epoch = train(modellist, cfg.experiment.lr, cfg.experiment.num_epochs, train_dataloader, eval_dataloader, device, cfg)
        
    test_MSE, test_rmse, test_nll = regression_evaluate_modellist(modellist, dataloader=test_dataloader, device=device, config=cfg)

    print(f"Test MSE: {test_MSE:.4f}, Test RMSE: {test_rmse:.4f}, Test  NLL: {test_nll:.4f}, Avg Time / Epoch: {avg_train_time_per_epoch:.4f} ")

    
    #plot 1D regression datasets
    if cfg.task.dataset in  ["sine", "gap"]:
        # Constructing the save path
        save_path = "debug/" + cfg.task.dataset
        plot_name = cfg.experiment.method
        
        if cfg.experiment.method == "SVN":
            # Assuming you have a way to specify or retrieve the type of Hessian calculation
            # For demonstration, let's say it's another configuration parameter under experiment
            hessian_type = cfg.SVN.hessian_calc  # This should be defined in your config
            plot_name += f"_n_epchs{cfg.experiment.num_epochs}_n_samp{cfg.experiment.n_samples}_{hessian_type}_lr{cfg.experiment.lr}_parts{cfg.experiment.n_particles}.png"
        else:
            plot_name += f"_n_epchs{cfg.experiment.num_epochs}_n_samp{cfg.experiment.n_samples}_lr{cfg.experiment.lr}_parts{cfg.experiment.n_particles}.png"
        
        full_save_path = f"{save_path}/{plot_name}"
        print("Save path:", full_save_path)
    
        plot_modellist(modellist, x_train, y_train, x_test, y_test, full_save_path)
