# TODO 单个group用features MS
# TODO test的逻辑


for i in 1 2 3 4 5 6 7 8 9 10
do
  DATA_PATH=adjusted_data_group_${i}.csv
  MODEL_ID=custom_group_${i}
  python3 run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --model_id $MODEL_ID \
    --model TimesNet \
    --data custom \
    --root_path ./datasets \
    --data_path $DATA_PATH \
    --features M \
    --freq s \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 1 \
    --inverse \
    --train_epochs 5 \
    --group $i \
    --batch_size 32 \ 
done  # <--- 这里必须有 done
