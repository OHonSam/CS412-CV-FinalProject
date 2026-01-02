set -x

export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_0,mlx5_2
export TRITON_CACHE_DIR="/tmp/triton"
export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=work_dirs/${JOB_NAME}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

PARTITION=${PARTITION:-"video"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"auto"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-1}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-4}


MAX_LENGHT=8192
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
   python -u unify_internvl2_train_r16.py \
  --model OpenGVLab/InternVL_2_5_HiCo_R16 \
  --datasets data/diy_ft_data.json \
  --num-workers 4 \
  --mirco-batch-size $MIRCO_BATCH_SIZE \
  --global-batch-size $((MIRCO_BATCH_SIZE*GPUS*ACCUMULATIVE_COUNTS)) \
  --vit_lr 2e-6 \
  --connector_lr 1e-5 \
  --lr 1e-5 \
  --wd 0.0 \
  --use-fast-tokenizer \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 2 \
  --seed 42 \
  --checkpoint-interval 4000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  --dset-pack \
  --dset-cache-dir ./internvl2_5_8b_sft_pack.sh_cache_r16_new_soha_video_vcg_c64 \
  --mirco-batch-size 1 \
  --max-length $MAX_LENGHT \
  --pack-max-length $((MIRCO_BATCH_SIZE * MAX_LENGHT)) \
  --concat-before-pack \
  --group-by-length \
  --min_num_frames 64 \
  --max_num_frames 256 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
