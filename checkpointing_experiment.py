import os
import re
import datetime 
import wandb

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from models.mlp import initialise_mlp_models
from models.lenet import initialise_lenet_models
from models.wrn import initialise_resnet32_modellist

from utils.kernel import RBF
from utils.data import get_sine_data, get_gap_data, load_yacht_data, \
    load_energy_data, load_autompg_data, load_concrete_data, load_kin8nm_data, \
        load_protein_data, load_naval_data, load_power_data, load_parkinson_data, \
        load_mnist_data, load_fashionmnist_data, load_breast_data, load_heart_data, \
        load_ionosphere_data, load_australian_data, load_cifar10_data
from utils.plot import plot_modellist
from utils.eval import regression_evaluate_modellist, classification_evaluate_modellist
from train.train import train
from utils.experiment_utils import Dataset, generate_model_path

def increment_checkpoint_path(original_path):
    # Extract the base directory and file name
    base_dir, filename = os.path.split(original_path)
    
    # Use a regex to find the integer 'n' in the filename
    match = re.search(r'checkpoint_(\d+)\.pt', filename)
    

    if match:
        n = int(match.group(1))  # Get the integer n
        new_n = n + 1  # Increment n by 1
        # Create the new filename by replacing n with n+1
        #new_filename = filename.replace(f'checkpoint_{n}.pt', f'checkpoint_{new_n}.pt')
        new_filename = re.sub(r'checkpoint_\d+\.pt', f'checkpoint_{new_n}.pt', filename)
        # Combine the base directory with the new filename
        print(new_filename)
        
    else:
        # If there is no match, assume no checkpoint number and append one
        new_filename = filename.replace('.pt', '') + '_checkpoint_0.pt'  # Remove '.pt' and add '_checkpoint_0.pt'
    
    # Combine the base directory with the new filename
    new_path = os.path.join(base_dir, new_filename)
    return new_path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_checkpointing_experiment(cfg):


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
    elif cfg.task.dataset =="breast":
        x_train, y_train, x_test, y_test = load_breast_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="heart":
        x_train, y_train, x_test, y_test = load_heart_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="ionosphere":
        x_train, y_train, x_test, y_test = load_ionosphere_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="australian":
        x_train, y_train, x_test, y_test = load_australian_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="mnist":
        x_train, y_train, x_test, y_test = load_mnist_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="fashionmnist":
        x_train, y_train, x_test, y_test = load_fashionmnist_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="cifar10":
        x_train, y_train, x_test, y_test = load_cifar10_data(config=cfg)
    else: 
        print('The configured dataset is not yet implemented')
        ValueError("The configured dataset is not yet implemented")
        return
    
    method = cfg.experiment.method
    task = cfg.task.dataset

    if method == 'SVN':
            active_tags = [method, task, cfg.SVN.hessian_calc]
    else:
        active_tags = [method, task]
            
    wandb_group = cfg.experiment.wandb_group
    wandb.init( project="SVN_Ensembles", 
                tags=active_tags,
                entity="klemens-floege",
                group=wandb_group
            )
     # Setting the configuration in WandB
    if cfg.experiment.method == 'SVN': 
        wandb.config.update({
            "learning_rate": cfg.experiment.lr,
            "num_epochs": cfg.experiment.num_epochs,
            "batch_size": cfg.experiment.batch_size,
            "early_stopping": cfg.experiment.early_stopping,
            "dataset": cfg.task.dataset,
            "method": cfg.experiment.method,
            "task_type": cfg.task.task_type,
            "n_splits": cfg.experiment.n_splits, 
            "hessian_calc": cfg.SVN.hessian_calc,
            "use_curvature_kernel": cfg.SVN.use_curvature_kernel
        })
    else: 
        wandb.config.update({
            "learning_rate": cfg.experiment.lr,
            "num_epochs": cfg.experiment.num_epochs,
            "batch_size": cfg.experiment.batch_size,
            "early_stopping": cfg.experiment.early_stopping,
            "dataset": cfg.task.dataset,
            "method": cfg.experiment.method,
            "task_type": cfg.task.task_type,
            "n_splits": cfg.experiment.n_splits
        })

    output_dim = cfg.task.dim_problem  # Assuming y_train is a vector; if it's a 2D array with one column, this is correct

    if cfg.task.dataset in ['mnist', 'fashionmnist']:
        image_dim = cfg.task.image_dim
        raw_modellist = initialise_lenet_models(image_dim, output_dim, cfg)
    elif cfg.task.dataset in ['cifar10']:
        raw_modellist = initialise_resnet32_modellist(depth =34, widen_factor=1 , config=cfg)
    else: 
        print('Checkpoiting only for Image dataset')
        ValueError("Checkpoiting only for Image dataset")
        return

    modellist = []  # This should be your logic to load models
    # Initialize or load model list based on config
    if cfg.Checkpointing.load_pretrained:
        print('start laoding')
        fold = 0
        #model_path = 'mnist/Ensemble/batch16_ep4_lr0.03/2024-05-11_22-03-39'
        #model_path = 'mnist/Ensemble/batch16_ep4_lr0.03/2024-05-11_22-03-39/checkpoint_1.pt'
        model_path = cfg.Checkpointing.model_path
        #checkpoint_path = os.path.join(cfg.experiment.base_save_path, model_path,  f"model_fold{fold+1}.pt")
        checkpoint_path = os.path.join(cfg.experiment.base_save_path, model_path)
        
        if os.path.exists(checkpoint_path):
            print(f"Loading models from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)

            print('state dict keys', state_dict.keys())
            
            for i in range(cfg.experiment.n_particles):  # Adjust the range based on how many models you expect
                model = raw_modellist[i]  # This should be your function to create a new model instance
                model.load_state_dict(state_dict[f'model_{i}'])
                modellist.append(model)
        else: 
            print('checkpoint path does not exxit')
            ValueError('checkpoint path does not exxit')
            return
    
    print('modellist loaded')
    print(modellist)

    
    # Split the data into training and evaluation sets
    x_train_split, x_eval_split, y_train_split, y_eval_split = train_test_split(x_train, y_train, test_size=cfg.experiment.train_val_split, random_state=cfg.experiment.seed)
    
    # Create instances of the SineDataset for each set
    train_dataset = Dataset(x_train_split, y_train_split)
    eval_dataset = Dataset(x_eval_split, y_eval_split)
    test_dataset = Dataset(x_test, y_test)
    
    print("length train", len(train_dataset))
    print("length eval", len(eval_dataset))
    print("length test", len(test_dataset))
    
    #create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
    
    

    avg_train_time_per_epoch = train(modellist, cfg.experiment.lr, cfg.experiment.num_epochs, train_dataloader, eval_dataloader, device, cfg)
        
    if cfg.task.task_type == 'regression':
            test_MSE, test_rmse, test_nll = regression_evaluate_modellist(modellist, dataloader=test_dataloader, device=device, config=cfg)
            print(f"Test MSE: {test_MSE:.4f}, Test RMSE: {test_rmse:.4f}, Test  NLL: {test_nll:.4f}, Avg Time / Epoch: {avg_train_time_per_epoch:.4f} ")
            # Log regression test metrics
            wandb.run.summary.update({
                "test_MSE": test_MSE,
                "test_RMSE": test_rmse,
                "test_NLL": test_nll,
                "average_train_time_per_epoch": avg_train_time_per_epoch
            })
    elif cfg.task.task_type == 'classification':
        test_accuracy, test_cross_entropy, test_entropy, test_nll, test_ece, test_brier, test_AUROC = classification_evaluate_modellist(modellist, dataloader=test_dataloader, device=device, config=cfg)
        print(f"Test Acc: {test_accuracy:.4f}, Test CrossEntropy: {test_cross_entropy:.4f}, Test Entropy: {test_entropy:.4f}, Test NLL: {test_nll:.4f}, Test ECE: {test_ece:.4f}, Test Brier: {test_brier:.4f}, Test AUROC: {test_AUROC:.4f}, Avg Time / Epoch: {avg_train_time_per_epoch:.4f} ")
        # Log classification test metrics
        wandb.run.summary.update({
            "test_accuracy": test_accuracy,
            "test_cross_entropy": test_cross_entropy,
            "test_entropy": test_entropy,
            "test_NLL": test_nll,
            "test_ECE": test_ece,
            "test_Brier": test_brier,
            "test_AUROC": test_AUROC,
            "average_train_time_per_epoch": avg_train_time_per_epoch
        })

     # Save model checkpoint if required
    if cfg.experiment.save_model:
        #modellist_path = '/Users/klemens.floege/Desktop/dev/laplace_SVN/model_checkpoints/mnist/Ensemble/batch16_ep4_lr0.03/2024-05-11_22-03-39'
        
        #modellist_path = os.path.join(base_save_path, model_path)
        print('Saving the model')
        
        #modellist_path = increment_checkpoint_path(checkpoint_path)
        #base_dir, cpt_string = increment_checkpoint_path(checkpoint_path)
        save_path = increment_checkpoint_path(checkpoint_path)
        print(save_path)
        #print(base_dir)
 
        #if not os.path.exists(modellist_path):
        #    os.makedirs(modellist_path)
        
        #save_path = os.path.join(base_dir, cpt_string)

        
        combined_state_dict = {f'model_{i}': model.state_dict() for i, model in enumerate(modellist)}

        torch.save(combined_state_dict, save_path)
        print(f"Combined model state dictionary saved to {save_path}")
         
        

    # Ensure avg_train_time_per_epoch is a tensor and move to CPU
    if isinstance(avg_train_time_per_epoch, torch.Tensor):
        avg_train_time_per_epoch = avg_train_time_per_epoch.cpu()

    wandb.finish()
    print('finish')