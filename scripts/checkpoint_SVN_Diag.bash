#!/bin/bash

# Define variables
NUM_EPOCHS=6
METHOD="SVN"
LR=3e-3
BATCH_SIZE=128
TASK="fashionmnist"
JOB_NAME="SVGD_pretrain_no_leak_${TASK}_${METHOD}_${NUM_EPOCHS}Ep_${LR}lr_${BATCH_SIZE}_fold5"

# Command to submit the job
SUBMIT_CMD="sbatch --nodes=1 --ntasks=1 --time=8:00:00 --partition=normal --gres=gpu:4 \
--output=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.out \
--error=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.err \
-J \"$JOB_NAME\" \
--wrap=\"source ~/.bash_profile; source myEnv/bin/activate; \
srun python checkpointing_main.py task='$TASK'  \
experiment.method='$METHOD' experiment.num_epochs=$NUM_EPOCHS experiment.lr=$LR \
SVN.use_curvature_kernel="use_curvature"  \
SVN.hessian_calc='Diag' \
experiment.save_model=True \
SVN.block_diag_approx=True \
Checkpointing.load_pretrained=True \
Checkpointing.model_path='fashionmnist/SVGD/batch128_ep50_lr0.003/2024-05-14_01-09-01/model_fold5.pt' \
experiment.batch_size=$BATCH_SIZE\""

# Execute the submission command
echo "Submitting job: $JOB_NAME"
eval $SUBMIT_CMD