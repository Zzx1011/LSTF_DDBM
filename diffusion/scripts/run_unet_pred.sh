#!/bin/bash
#SBATCH --account=rrg-timsbc
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G              # memory per node
#SBATCH --time=0-19:00
source /home/zzx/projects/rrg-timsbc/zzx/bin/activate
module load mpi4py/3.1.4 
module load python/3.11
module load scipy-stack
cd /home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion
echo "Running on $(date +%Y-%m-%d_%H-%M-%S) with PID $$"

python ./scripts/unet_pred.py 