#!/bin/bash

# Define variables
NUM_EPOCHS=1
METHOD="Ensemble"
LR=3e-2
BATCH_SIZE=32
DATASET="yacht"
JOB_NAME="${DATASET}_${METHOD}_${NUM_EPOCHS}e_${LR}_${BATCH_SIZE}"

# Path settings
OUT_PATH="/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/results"
JOB_FILE="${OUT_PATH}/${JOB_NAME}.sh"

# Check if output directory exists
if [ ! -d "$OUT_PATH" ]; then
    mkdir -p "$OUT_PATH"
    echo "Directory created: $OUT_PATH"
fi

# Generate job script
cat > $JOB_FILE <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --output=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.out
#SBATCH --error=/home/hgf_hmgu/hgf_tfv0045/svn_ensembles/outputs/%x.err
#SBATCH -J $JOB_NAME

source ~/.bash_profile
source myEnv/bin/activate

# Run Python script
srun python main.py experiment.dataset="$DATASET" \
 experiment.method="$METHOD" experiment.num_epochs=$NUM_EPOCHS experiment.lr=$LR \
 experiment.batch_size=$BATCH_SIZE
EOF

# Submit the job
sbatch $JOB_FILE
