import os
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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def pretrain_experiment(cfg):
    # Load your dataset here
    print(cfg.task.dataset)

    
    if cfg.task.dataset =="mnist":
        x_train, y_train, x_test, y_test = load_mnist_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="fashionmnist":
        x_train, y_train, x_test, y_test = load_fashionmnist_data(test_size_split=cfg.experiment.train_val_split, seed=cfg.experiment.seed, config=cfg)
    elif cfg.task.dataset =="cifar10":
        x_train, y_train, x_test, y_test = load_cifar10_data(config=cfg)
    else: 
        print('The configured dataset is not yet implemented for pre-training')
        ValueError("The configured dataset is not yet implemented for pre-training")
        return
    

    n_splits = cfg.experiment.n_splits  # Number of folds for k-fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.experiment.seed)


    if cfg.task.task_type == 'regression':
        n_metrics = 3 # metrics to track across folds
    elif cfg.task.task_type == 'classification':
        n_metrics = 7 # metrics to track across folds

    metrics_array = np.zeros((n_splits, n_metrics)) # Store metrics from each fold
    model_path = generate_model_path(cfg)
    
    #jsut do CV for train,test to prevent leakage
    for fold, (train_idx, valid_idx) in enumerate(kf.split(x_train)):
        print(f"Running fold {fold + 1}/{n_splits}")

        method = cfg.experiment.method
        task = cfg.task.dataset
        #wandb_group = cfg.experiment.wandb_group
 
        if method == 'SVN':
            active_tags = [f"Fold_{fold+1}", method, task, cfg.SVN.hessian_calc]
        else:
            active_tags = [method, f"Fold_{fold+1}", task]
            
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
        
        # Split data into training and validation for this fold
        x_train_fold, x_valid_fold = x_train[train_idx], x_train[valid_idx]
        y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]



        #initialise models and normalise data
        output_dim = cfg.task.dim_problem  # Assuming y_train is a vector; if it's a 2D array with one column, this is correct

        if cfg.task.dataset in ['mnist', 'fashionmnist']:
            image_dim = cfg.task.image_dim
            modellist = initialise_lenet_models(image_dim, output_dim, cfg)
        elif cfg.task.dataset in ['cifar10']:
            modellist = initialise_resnet32_modellist(depth =34, widen_factor=1 , config=cfg)
        
        

        # Create instances of the SineDataset for each set
        train_dataset = Dataset(x_train_fold, y_train_fold)
        eval_dataset = Dataset(x_valid_fold, y_valid_fold)
        test_dataset = Dataset(x_test, y_test)

        print("length train", len(train_dataset))
        print("length eval", len(eval_dataset))
        print("length test", len(test_dataset))
        
        #create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
        eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.experiment.batch_size, shuffle=cfg.experiment.shuffle)
        
        
        # Train and evaluate as before
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
            if isinstance(test_ece, torch.Tensor):
                test_ece = test_ece.cpu()
            if isinstance(test_brier, torch.Tensor):
                test_brier = test_brier.cpu()
            if isinstance(test_AUROC, torch.Tensor):
                test_AUROC = test_AUROC.cpu()
            metrics_array[fold] = [test_accuracy,test_cross_entropy,  test_nll, test_ece, test_brier,test_AUROC, avg_train_time_per_epoch]
        

        # Save model checkpoint if required
        if cfg.experiment.save_model:
            base_save_path = cfg.experiment.base_save_path
            
            modellist_path = os.path.join(base_save_path, model_path)

            # Ensure the directory exists
            if not os.path.exists(modellist_path):
                os.makedirs(modellist_path)

            save_path = os.path.join(modellist_path, f"model_fold{fold+1}.pt")
            combined_state_dict = {f'model_{i}': model.state_dict() for i, model in enumerate(modellist)}
    
            torch.save(combined_state_dict, save_path)
            print(f"Combined model state dictionary saved to {save_path}")
            

        wandb.finish()


    # Calculate mean and standard deviation for each metric across all folds
    metrics_mean = metrics_array.mean(axis=0)
    metrics_std = metrics_array.std(axis=0) / np.sqrt(cfg.experiment.n_splits)


    if cfg.task.task_type == 'regression':
        print(f"Average Test MSE: {metrics_mean[0]:.2f} ± {metrics_std[0]:.2f}, Average Test NLL: {metrics_mean[1]:.2f} ± {metrics_std[1]:.2f},  Avg Time / Epoch: {metrics_mean[2]:.2f} ± {metrics_std[2]:.2f}")
    elif cfg.task.task_type == 'classification':
        print(f"Avg Test Accuracy: {metrics_mean[0]:.2f} ± {metrics_std[0]:.2f}, Avg Test CrossEntr: {metrics_mean[1]:.2f} ± {metrics_std[1]:.2f}, Avg NLL: {metrics_mean[2]:.2f} ± {metrics_std[2]:.2f}, Avg Test ECE: {metrics_mean[3]:.2f} ± {metrics_std[3]:.2f}, Avg Test Brier: {metrics_mean[4]:.2f} ± {metrics_std[4]:.2f},  Avg AUROC: {metrics_mean[5]:.2f} ± {metrics_std[5]:.2f}, Avg Time / Epoch: {metrics_mean[6]:.2f} ± {metrics_std[6]:.2f}")


    
    print('finish')