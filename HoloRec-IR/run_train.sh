# export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,3
export HF_ENDPOINT=https://hf-mirror.com

DATASET=Beauty
OUTPUT_DIR=./ckpt/origin_beauty_0.5_1_30_same/

torchrun --nproc_per_node=2 --master_port=4314 ./finetune.py \
    --base_model t5-small \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --wandb_run_name origin_beauty_0.5_1_30_new \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 400 \
    --index_file .tw32.json \
    --coarse_index_file .tw8.json \
    --temperature 1.0 \
    --coarse_loss_weight 0.5 \
    --fine_loss_weight 1.0 \
    --coarse_align_weight 30.0

