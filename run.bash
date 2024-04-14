#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -J concrete_3e-2_50_SVN
#SBATCH --output=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.out
#SBATCH --error=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.err
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4


# Your other commands
CODE_PATH="/home/hgf_hmgu/hgf_tfv0045/svn_ensembles"
OUT_PATH="/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/results"

# Check if the directory exists
if [ ! -d "$OUT_PATH" ]; then
    # Directory does not exist, so create it
    mkdir -p "$OUT_PATH"
    echo "Directory created: $OUT_PATH"
else
    echo "Directory already exists: $OUT_PATH"
fi

source ~/.bash_profile


# Activate your environment
source myEnv/bin/activate


# Run your Python script
srun python main.py experiment.dataset="yacht" \
 experiment.method="SVN" experiment.num_epochs=50 experiment.lr=3e-2 
