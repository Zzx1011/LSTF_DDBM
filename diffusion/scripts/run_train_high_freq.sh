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

FREQ_SAVE_ITER=20000
NGPU=1
IMG_SIZE=256
NUM_CH=256
ATTN=32,16,8
EXP="ett${IMG_SIZE}_${NUM_CH}d_high_freq_$(date +%Y-%m-%d_%H-%M-%S)"
BS=64 #check this
DATASET=ett
DATA_DIR=/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/data/ETT
PRED=vp
NUM_RES_BLOCKS=3
COND=concat
SAMPLER=real-uniform
USE_16FP=True
ATTN_TYPE=flash
BETA_D=2
BETA_MIN=0.1
SIGMA_MAX=1
SIGMA_MIN=0.0001
SIGMA_DATA=0.5
COV_XY=0
FREQ_SAVE_ITER=20000
SAVE_ITER=100000



python ./scripts/ddbm_train_high_freq.py --exp=$EXP \
 --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
  --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
   --image_size $IMG_SIZE --lr 0.0001 --num_channels $NUM_CH --num_head_channels 64 \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
     --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
      --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
      --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
      --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
      ${CKPT:+ --resume_checkpoint="${CKPT}"} 