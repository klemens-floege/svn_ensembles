
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from models.mlp import initliase_mlp_models

from utils.kernel import RBF
from utils.data import get_sine_data, get_gap_data, load_yacht_data, \
    load_energy_data, load_autompg_data, load_concrete_data, load_kin8nm_data, \
        load_protein_data, load_naval_data, load_power_data, load_parkinson_data
from utils.plot import plot_modellist
from utils.eval import regression_evaluate_modellist, classification_evaluate_modellist
from train.train import train

from sklearn.preprocessing import StandardScaler

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
    # Load your dataset here
    print(cfg.task.dataset)

    if cfg.task.dataset == "sine":
        x_train, y_train, x_test, y_test = get_sine_data(n_samples=cfg.experiment.n_samples, seed= cfg.experiment.seed)
    elif cfg.task.dataset == "gap":
        x_train, y_train, x_test, y_test = get_gap_data(n_samples=cfg.experiment.n_samples, seed= cfg.experiment.seed)
    elif cfg.task.dataset == "yacht":
        x_train, y_train, x_test, y_test = load_yacht_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset == "energy":
        x_train, y_train, x_test, y_test = load_energy_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset == "autompg":
        x_train, y_train, x_test, y_test = load_autompg_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="concrete":
        x_train, y_train, x_test, y_test = load_concrete_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="kin8nm":
        x_train, y_train, x_test, y_test = load_kin8nm_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="naval":
        x_train, y_train, x_test, y_test = load_naval_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="protein":
        x_train, y_train, x_test, y_test = load_protein_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="power":
        x_train, y_train, x_test, y_test = load_power_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="parkinsons":
        x_train, y_train, x_test, y_test = load_parkinson_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    else: 
        print('The configured dataset is not yet implemented')
        ValueError("The configured dataset is not yet implemented")
        return

    n_splits = cfg.experiment.n_splits  # Number of folds for k-fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.experiment.seed)
    
    x_combined = np.concatenate((x_train, x_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)

    if cfg.task.task_type == 'regression':
        n_metrics = 3 # metrics to track across folds
    elif cfg.task.task_type == 'classification':
        n_metrics = 4 # metrics to track across folds

    metrics_array = np.zeros((n_splits, n_metrics)) # Store metrics from each fold
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_combined)):
        print(f"Running fold {fold + 1}/{n_splits}")
        
        # Split data into training and validation for this fold
        x_train_fold, x_test_fold = x_combined[train_idx], x_combined[test_idx]
        y_train_fold, y_test_fold = y_combined[train_idx], y_combined[test_idx]
 
        scaler = StandardScaler()
        x_train_fold = scaler.fit_transform(x_train_fold)
        x_test_fold = scaler.transform(x_test_fold)

         # Split the training data into training and evaluation sets
        #x_train_split, x_eval_split, y_train_split, y_eval_split = train_test_split(x_train, y_train, test_size=cfg.experiment.train_test_split, random_state=cfg.experiment.seed)
        x_train_split, x_eval_split, y_train_split, y_eval_split = train_test_split(x_train_fold, y_train_fold, test_size=cfg.experiment.train_val_split, random_state=cfg.experiment.seed)

        
        # Create instances of the SineDataset for each set
        train_dataset = Dataset(x_train_split, y_train_split)
        eval_dataset = Dataset(x_eval_split, y_eval_split)
        #test_dataset = Dataset(x_test, y_test)
        test_dataset = Dataset(x_test_fold, y_test_fold)

        print("length train", len(train_dataset))
        print("length eval", len(eval_dataset))
        print("length test", len(test_dataset))
        
        

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
        eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
        
        # Initialize your models for this fold
        
        
        print("x_Train shape: ", x_train.shape)
        input_dim = x_train.shape[1]  # Number of features in x_train
        #input_dim = 1 # Number of features in x_train
        output_dim = cfg.task.dim_problem  # Assuming y_train is a vector; if it's a 2D array with one column, this is correct
        #print(y_train.shape)
        modellist = initliase_mlp_models(input_dim, output_dim, cfg)
        
        
        # Train and evaluate as before
        avg_train_time_per_epoch = train(modellist, cfg.experiment.lr, cfg.experiment.num_epochs, train_dataloader, eval_dataloader, device, cfg)
        
        #test_MSE, test_rmse, test_nll = evaluate_modellist(modellist, dataloader=test_dataloader, device=device)
        #metric1, metric2, metric3 = None, None, None

        if cfg.task.task_type == 'regression':
            test_MSE, test_rmse, test_nll = regression_evaluate_modellist(modellist, dataloader=eval_dataloader, device=device, config=cfg)
            metric1, metric2, metric3 =  test_MSE, test_rmse, test_nll
            print(f"Test MSE: {test_MSE:.4f}, Test RMSE: {test_rmse:.4f}, Test  NLL: {test_nll:.4f}, Avg Time / Epoch: {avg_train_time_per_epoch:.4f} ")
        elif cfg.task.task_type == 'classification':
            test_accuracy, test_cross_entropy, test_entropy, test_nll = classification_evaluate_modellist(modellist, dataloader=eval_dataloader, device=device, config=cfg)
            metric1, metric2, metric3 = test_cross_entropy, test_entropy, test_nll
            print(f"Acc: {test_accuracy:.4f}, Test CrossEntropy: {test_cross_entropy:.4f}, Test  Entropy: {test_entropy:.4f}, Test  NLL: {test_nll:.4f}, Avg Time / Epoch: {avg_train_time_per_epoch:.4f} ")
         
        
    
        
        

        # Ensure avg_train_time_per_epoch is a tensor and move to CPU
        if isinstance(avg_train_time_per_epoch, torch.Tensor):
            avg_train_time_per_epoch = avg_train_time_per_epoch.cpu()

        #metrics_array[fold] = [test_MSE, test_nll, avg_train_time_per_epoch]
        if cfg.task.task_type == 'regression':
            if isinstance(test_MSE, torch.Tensor):
                test_MSE = test_MSE.cpu()
            if isinstance(test_nll, torch.Tensor):
                test_nll = test_nll.cpu()
            metrics_array[fold] = [test_MSE, test_nll, avg_train_time_per_epoch]

        elif cfg.task.task_type == 'classification':
            if isinstance(test_accuracy, torch.Tensor):
                test_accuracy = test_accuracy.cpu()
            if isinstance(test_cross_entropy, torch.Tensor):
                test_cross_entropy = test_cross_entropy.cpu()
            if isinstance(test_nll, torch.Tensor):
                test_nll = test_nll.cpu()
            metrics_array[fold] = [test_accuracy,test_cross_entropy,  test_nll, avg_train_time_per_epoch]
        

        if fold== 0 and cfg.task.dataset in  ["sine", "gap"]: 
             # Constructing the save path
            save_path = "images/" + cfg.task.dataset
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


    
    # After all folds are complete, aggregate your metrics across folds to evaluate overall performance
    #aggregate_metrics(metrics_list)
    # Convert metrics_list to a numpy array for easier calculation of mean and std
    #metrics_array = np.array(metrics_list)
    #metrics_array = np.array([metric.detach().cpu() for metric in metrics_list])


    # Calculate mean and standard deviation for each metric across all folds
    metrics_mean = metrics_array.mean(axis=0)
    metrics_std = metrics_array.std(axis=0)

    if cfg.task.task_type == 'regression':
        print(f"Average Test MSE: {metrics_mean[0]:.2f} ± {metrics_std[0]:.2f}, Average Test NLL: {metrics_mean[1]:.2f} ± {metrics_std[1]:.2f},  Avg Time / Epoch: {metrics_mean[2]:.2f} ± {metrics_std[2]:.2f}")
    elif cfg.task.task_type == 'classification':
        print(f"Avg Test Accuracy: {metrics_mean[0]:.2f} ± {metrics_std[0]:.2f}, Avg Test CrossEntr: {metrics_mean[1]:.2f} ± {metrics_std[1]:.2f}, Avg NLL: {metrics_mean[2]:.2f} ± {metrics_std[2]:.2f},  Avg Time / Epoch: {metrics_mean[3]:.2f} ± {metrics_std[3]:.2f}")
    