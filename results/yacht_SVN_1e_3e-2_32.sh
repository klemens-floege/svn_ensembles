#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --output=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.out
#SBATCH --error=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.err
#SBATCH -J yacht_SVN_1e_3e-2_32

source ~/.bash_profile
source myEnv/bin/activate

# Run Python script
srun python main.py experiment.dataset="yacht"  experiment.method="SVN" experiment.num_epochs=1 experiment.lr=3e-2  experiment.batch_size=32
