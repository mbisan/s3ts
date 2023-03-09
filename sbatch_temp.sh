#!/bin/bash
#SBATCH --job-name=ResNet_TS_quant_CBF
#SBATCH --acount=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rcoterillo@bcamath.org
#SBATCH -o /home/rcoterillo/repos/s3ts/outputs/ResNet_TS_quant_CBF.out
#SBATCH -e /home/rcoterillo/repos/s3ts/outputs/ResNet_TS_quant_CBF.err
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Python/3.10.4-GCCcore-11.3.0
source /scratch/rcoterillo/s3ts/s3ts_env/bin/activate
python /scratch/rcoterillo/s3ts/main_cli.py --dataset CBF --mode TS --arch ResNet --exp quant