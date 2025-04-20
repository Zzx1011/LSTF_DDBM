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
# pip install einops
# pip install piq
# pip install blobfile
# pip install torchvision

echo "Running on $(date +%Y-%m-%d_%H-%M-%S) with PID $$"
EXP+="_vp"
COND=concat
BETA_D=2
BETA_MIN=0.1
SIGMA_MAX=1
SIGMA_MIN=0.0001
COV_XY=0
python ./scripts/image_sample.py --num_head_channels 64 --pred_mode=vp --batch_size 16 --churn_step_ratio 0.5 --steps 40 \
--sampler heun --num_res_blocks 3 --image_size 64 --num_channels 192 --exp=$EXP ${COND:+ --condition_mode="${COND}"} \
--sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} --use_fp16 True \
--cov_xy $COV_XY --class_cond False --split test --guidance 1 --use_scale_shift_norm True --weight_schedule bridge_karras \
--rho 7 --upscale=False --dropout 0.1 --attention_type flash --attention_resolutions 32,16,8 --resblock_updown True
