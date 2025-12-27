#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# === 1. SET CUDA ENVIRONMENT ===
# Pointing to your local llava_env where nvcc was found
export CUDA_HOME=/datastore/clc_hcmus/ZaAIC/envs/videochat_light
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# === 2. SETUP PATHS ===
# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CODE_BASE="$SCRIPT_DIR/VideoChat-Flash"
DATA_PATH="$SCRIPT_DIR/sutd_llava_train.json"
VIDEO_FOLDER="$SCRIPT_DIR/SUTD/videos/"
OUTPUT_DIR="$SCRIPT_DIR/finetune/checkpoints/videochat-2b-sutd-lora"

# === 3. CHECK & RUN ===
if [ ! -d "$CODE_BASE" ]; then
    echo "Error: LLaVA-NeXT folder not found. Run: git clone https://github.com/LLaVA-VL/LLaVA-NeXT"
    exit 1
fi

cd "$CODE_BASE"

# Using python -m deepspeed.launcher.runner to invoke the specific deepspeed in your env
python3 -m deepspeed.launcher.runner --num_gpus=1 llava-train_videochat/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448 \
    --version qwen_2_5 \
    --data_path "$DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --frames_upbound 32 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard