#!/bin/bash

# Define variables
NUM_EPOCHS=5
METHOD="SVGD"
LR=3e-3
BATCH_SIZE=128
TASK="cifar10"
JOB_NAME="${TASK}_pretraining_${METHOD}_${NUM_EPOCHS}Ep_${LR}lr_${BATCH_SIZE}Bsz"

# Command to submit the job
SUBMIT_CMD="sbatch --nodes=1 --ntasks=1 --time=8:00:00 --partition=normal --gres=gpu:4 \
--output=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.out \
--error=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.err \
-J \"$JOB_NAME\" \
--wrap=\"source ~/.bash_profile; source myEnv/bin/activate; \
srun python pretrain_main.py task='$TASK'  \
experiment.method='$METHOD' experiment.num_epochs=$NUM_EPOCHS experiment.lr=$LR \
SVN.use_curvature_kernel="use_curvature"  \
SVN.hessian_calc='Diag' \
experiment.save_model=True \
SVN.block_diag_approx=True \
experiment.batch_size=$BATCH_SIZE\""

# Execute the submission command
echo "Submitting job: $JOB_NAME"
eval $SUBMIT_CMD