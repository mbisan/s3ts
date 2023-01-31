#!/bin/bash

#SBATCH --job-name=S3TS_test
#SBATCH --partition=large   # large / gpu
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rcoterillo@bcamath.org
#SBATCH -o %x-%a.out
#SBATCH -e %x-%a.err
#SBATCH -C bdw

# CDIR=`pwd`

# echo $SCRATCH_DIR
# echo $SLURM_JOB_ID
# echo $SLURM_ARRAY_JOB_ID
# echo "-------"
# cp ./* -v $SCRATCH_DIR
# cd $SCRATCH_DIR

echo "-------"

# echo "Slurm job id is ${SLURM_JOB_ID}"
# echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
# echo "Instance index is ${SLURM_ARRAY_TASK_ID}."
# echo "-------"

# Load software
echo "Loading modules"
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1
module laod cuDNN/8.2.1.32-CUDA-11.3.1
echo "Modules loaded"
echo "Activating environment"
source $HOME/s3ts/envs/s3ts_env/bin/activate
echo "Environment ready"
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
