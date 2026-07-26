DATASET=Beauty
DATA_PATH=../data
MODEL_NAME=${DATASET}-tw32-0.5-1-30
CKPT_PATH=./ckpt/${MODEL_NAME}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_FILE=./results_1/${MODEL_NAME}.json
SAVE_PRED_TXT=./results_1/${MODEL_NAME}-predictions.txt

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=2214 test.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --test_batch_size 32 \
  --num_beams 20 \
  --save_pred_txt $SAVE_PRED_TXT \
  --test_prompt_ids 0 \
  --index_file .tw32.json \
  --coarse_index_file .tw32.json