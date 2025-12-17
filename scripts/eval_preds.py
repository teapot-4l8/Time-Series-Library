#!/usr/bin/env python3
"""
Evaluate predictions produced by the short-term experiment.
- Loads forecasts CSV saved in `./m4_results/TimesNet/Monthly_forecast.csv`
- Aligns predictions with original dataset timestamps from `./datasets/adjusted_data_with_seconds.csv`
- Saves `results.csv` with columns: `date,true,pred,error,abs_error,sq_error`
- Plots: `true_vs_pred.png` and `error_over_time.png`
- Prints metrics: MAE, MSE, RMSE, MAPE, sMAPE

Usage:
  python3 scripts/eval_preds.py

Adjust constants below if your files are in different locations.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from math import sqrt

# Paths (adjust if needed)
FORECAST_CSV = './m4_results/TimesNet/Monthly_forecast.csv'
RAW_CSV = './datasets/adjusted_data_with_seconds.csv'
OUT_DIR = './evaluation_results'

SEQ_LEN = 12
LABEL_LEN = 6
PRED_LEN = 1
TARGET_COL = '10LBA10CT103K'
DATE_COL = 'date'

os.makedirs(OUT_DIR, exist_ok=True)

# load forecasts
if not os.path.exists(FORECAST_CSV):
    raise SystemExit(f'Forecast CSV not found: {FORECAST_CSV}')

forecasts_df = pd.read_csv(FORECAST_CSV, index_col=0)
# forecasts_df columns are V1..Vp where p == PRED_LEN
# flatten predictions to 1D array (assume pred_len == 1)
preds = forecasts_df.values.squeeze()

# load raw data
df = pd.read_csv(RAW_CSV)
# try parse date
try:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
except Exception:
    # try alternative formats
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=False, infer_datetime_format=True)

N = len(df)
num_train = int(N * 0.7)
num_test = int(N * 0.2)
num_val = N - num_train - num_test

border1s = [0, num_train - SEQ_LEN, N - num_test - SEQ_LEN]
border2s = [num_train, num_train + num_val, N]
# test partition
border1 = border1s[2]
border2 = border2s[2]

# number of samples in dataset.Data_x portion
data_len = border2 - border1
num_samples = data_len - SEQ_LEN - PRED_LEN + 1

if num_samples != preds.shape[0]:
    print(f'Warning: number of predictions ({preds.shape[0]}) does not match computed test samples ({num_samples}).')

results = []
for i in range(min(num_samples, preds.shape[0])):
    s_begin = i
    s_end = s_begin + SEQ_LEN
    r_begin = s_end - LABEL_LEN
    r_end = r_begin + LABEL_LEN + PRED_LEN
    # predicted timestamp(s): indices in original df = border1 + (r_begin + LABEL_LEN) .. border1 + (r_end - 1)
    pred_idx = border1 + r_begin + LABEL_LEN
    if isinstance(preds, np.ndarray):
        pred_val = float(preds[i])
    else:
        pred_val = float(preds[i])
    true_val = float(df[TARGET_COL].iloc[pred_idx])
    date_val = df[DATE_COL].iloc[pred_idx]
    error = pred_val - true_val
    results.append({'date': date_val, 'true': true_val, 'pred': pred_val, 'error': error, 'abs_error': abs(error), 'sq_error': error**2})

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, 'predictions_vs_true.csv'), index=False)

# metrics
mae = res_df['abs_error'].mean()
mse = res_df['sq_error'].mean()
rmse = sqrt(mse)
# MAPE: mean(|(pred-true)/true|) * 100 (avoid division by zero)
res_df['ape'] = res_df.apply(lambda r: abs(r['error'] / r['true']) if r['true'] != 0 else np.nan, axis=1)
mape = np.nanmean(res_df['ape']) * 100
# sMAPE
res_df['smape_term'] = res_df.apply(lambda r: 0 if (abs(r['pred']) + abs(r['true']))==0 else (2*abs(r['pred']-r['true']))/(abs(r['pred'])+abs(r['true'])), axis=1)
smape = np.nanmean(res_df['smape_term']) * 100

metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE(%)': mape, 'sMAPE(%)': smape}
print('Metrics:')
for k,v in metrics.items():
    print(f'  {k}: {v}')

# plots
plt.figure(figsize=(12,5))
plt.plot(res_df['date'], res_df['true'], label='True')
plt.plot(res_df['date'], res_df['pred'], label='Pred')
plt.xlabel('date')
plt.ylabel(TARGET_COL)
plt.legend()
plt.title('True vs Pred')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'true_vs_pred.png'))
plt.close()

# error over time
plt.figure(figsize=(12,4))
plt.plot(res_df['date'], res_df['error'], label='Error')
plt.xlabel('date')
plt.ylabel('Error (pred - true)')
plt.title('Prediction Error over Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'error_over_time.png'))
plt.close()

print(f'Wrote CSV and plots to {OUT_DIR}')

# also try to plot loss history if exists
# look for latest checkpoint folder
ckp_root = './checkpoints'
if os.path.exists(ckp_root):
    # find newest folder
    folders = [os.path.join(ckp_root,f) for f in os.listdir(ckp_root) if os.path.isdir(os.path.join(ckp_root,f))]
    if folders:
        latest = max(folders, key=os.path.getmtime)
        hist_path = os.path.join(latest, 'training_history.json')
        if os.path.exists(hist_path):
            hist = pd.read_json(hist_path)
            try:
                plt.figure()
                plt.plot(hist['train'], label='train')
                plt.plot(hist['val'], label='val')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.legend()
                plt.title('Training/Validation Loss')
                plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'))
                plt.close()
                print(f'Saved loss curve from {hist_path}')
            except Exception:
                pass

print('Done.')
