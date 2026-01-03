
# 自动读取 data_path 的最后一列作为 target
DATA_PATH=adjusted_data_group_7.csv
TARGET=$(head -1 ./datasets/$DATA_PATH | awk -F',' '{print $NF}')

python3 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id custom_group_7 \
  --model TimesNet \
  --data custom \
  --root_path ./datasets \
  --data_path $DATA_PATH \
  --features M \
  --target $TARGET \
  --freq s \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 1 \
  --enc_in 50 \
  --dec_in 50 \
  --c_out 50 \
  --inverse \
  --train_epochs 5 \
  --batch_size 32 \
  --devices 4 \

# python3 run.py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --model_id custom_group_7 \
#   --model TimesNet \
#   --data custom \
#   --root_path ./datasets \
#   --data_path adjusted_data_group_7.csv \
#   --features M \
#   --target 10CFA10GH001XQ41 \
#   --freq s \
#   --seq_len 12 \
#   --label_len 12 \
#   --pred_len 1 \
#   --enc_in 50 \
#   --dec_in 50 \
#   --c_out 50 \
#   --inverse \
#   --train_epochs 5 \
#   --batch_size 32 \
#   --devices 4 \