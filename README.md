# Stein Variational Newton Neural Network Ensembles

## Overview
This codebase is designed for the NeurIPS 2024 submission and implements the Stein Variational Newton (SVN) method along with other related methods. The supported methods and their configurations are detailed below.

## Supported Methods
- **SVN** (Stein Variational Newton)
- **LL-SVN** (Last-Layer Stein Variational Newton)
- **SVGD** (Stein Variational Gradient Descent)
- **Ensemble** (Deep Ensembles)
- **WGD** (Repulsive Ensembles)

## Hessian Approximations
The SVN method incorporates various Hessian approximations, each supporting multiple configurations.

### 1. Full Hessian
- **Full SVN Hessian**
- **Block Diagonal Approximation**
- **Normal Kernel**
- **Anisotropic Curvature Kernel**

### 2. KFAC Hessian
- **Full SVN Hessian**
- **Block Diagonal Approximation**
- **Normal Kernel**

### 3. Diagonal Hessian
- **Full Block Diagonal Approximation**
- **Normal Kernel**
- **Anisotropic Curvature Kernel**

## Running Experiments
We are using Hydra configs for configuration management. Please refer to the `config.yaml` file for further details. In order to call the LL-SVN algorithm simply set SVN.ll=True. 

### Example Command
To run the 'Yacht' dataset using the SVN Ensembles algorithm with the Block Diagonal approximation and using the curvature Kernel, use the following command:

```bash
python main.py task='parkinsons' experiment.num_epochs=1 experiment.wandb_logging=False experiment.method='SVN' SVN.hessian_calc='Diag' SVN.block_diag_approx=True


