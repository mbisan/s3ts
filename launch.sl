#!/bin/bash

#SBATCH --job-name=S3TS_test
#SBATCH --partition=bcam-exclusive
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=rcoterillo@bcamath.org
#SBATCH -o %x-%a.out
#SBATCH -e %x-%a.err

CDIR=`pwd`

# echo $SCRATCH_DIR
# echo $SLURM_JOB_ID
# echo $SLURM_ARRAY_JOB_ID
# echo "-------"
# cp ./* -v $SCRATCH_DIR
# cd $SCRATCH_DIR

echo "-------"

# Print job info
echo "Current directory is ${CDIR}"
echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."
echo "-------"

# Load software
echo "Loading modules and libraries..."
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/s3ts/s3ts_env/bin/activate
echo "Environment ready!"
echo "-------"

# check if file is script
PASSED=$1
if [[ -f $PASSED ]]; then
    echo "$PASSED seems to be a file, launching."
else
    echo "$PASSED is not a file, quitting."
    exit 1
fi

# launch script
python $PASSED
echo "Done"
