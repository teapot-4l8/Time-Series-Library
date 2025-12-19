#!/usr/bin/env bash
# Example script to run TimesNet short-term forecasting on a custom CSV
# Adjust paths and hyper-parameters as needed.

ROOT="./datasets"
DATA_FILE="adjusted_data_with_seconds.csv"
TARGET_COL="10LBA10CT103K"

# Number of input channels (auto-detect from CSV header: all columns except 'date')
# This computes enc_in/dec_in from the header row of the CSV file.
HEADER=$(head -n 1 "${ROOT}/${DATA_FILE}")
NUM_COLS=$(echo "$HEADER" | awk -F',' '{print NF}')
ENC_IN=$((NUM_COLS - 1))
DEC_IN=$ENC_IN
C_OUT=1

# Short-term settings (seconds frequency)
SEQ_LEN=12   # input length (seconds)
LABEL_LEN=6  # label / start length
PRED_LEN=1   # predict next 1 seconds

# Training options
EPOCHS=10
BATCH_SIZE=32
LR=0.0001

python3 run.py \
  --task_name short_term_forecast \
  --is_training 0 \
  --model_id custom_timesnet \
  --model TimesNet \
  --data custom \
  --root_path ${ROOT} \
  --data_path ${DATA_FILE} \
  --features MS \
  --target ${TARGET_COL} \
  --freq s \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --train_epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --des "custom_seconds" \
  --num_workers 4 \
  --gpu 0

echo "Run finished (check ./checkpoints and ./test_results)."
