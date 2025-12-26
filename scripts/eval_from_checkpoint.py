#!/usr/bin/env python3
"""
Evaluate a saved long-term checkpoint and produce metrics + plots.

Usage:
  python3 scripts/eval_from_checkpoint.py \
    --checkpoint /path/to/checkpoint.pth \
    --data_path ./datasets/adjusted_data_with_seconds.csv

Outputs:
  - ./results/<setting>/pred.npy and true.npy are saved by the experiment
  - ./evaluation_results/predictions_vs_true.csv
  - ./evaluation_results/true_vs_pred.png
  - ./evaluation_results/error_over_time.png

Fonts: plots use English Times New Roman.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
import sys
sys.path.append('.')
from types import SimpleNamespace
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def make_args(setting, data_path, seq_len=12, label_len=6, pred_len=6):
    # read data to infer channels
    df = pd.read_csv(data_path)
    cols = list(df.columns)
    if 'date' in cols:
        cols.remove('date')
    if '10LBA10CT103K' in cols:
        # keep target as last column per Dataset_Custom expectation
        pass
    enc_in = len(cols)

    args = SimpleNamespace()
    # basic
    args.task_name = 'long_term_forecast'
    args.is_training = 0
    args.model = 'TimesNet'
    args.model_id = 'custom'
    args.data = 'custom'
    args.features = 'MS'
    args.target = '10LBA10CT103K'
    args.freq = 's'
    args.checkpoints = './checkpoints'
    args.root_path = './datasets'
    args.data_path = os.path.basename(data_path)
    # model sizes (match training setting encoded in checkpoint folder)
    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.expand = 2
    args.d_conv = 4
    args.factor = 1
    args.embed = 'timeF'
    args.distil = True
    args.des = 'test'
    args.expand = 2
    args.d_conv = 4

    # IO / runtime
    args.batch_size = 32
    args.num_workers = 4
    args.use_gpu = torch.cuda.is_available()
    args.gpu_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0'
    args.inverse = True
    args.use_amp = False
    args.augmentation_ratio = 0
    args.seasonal_patterns = 'Monthly'
    args.use_dtw = False

    # TimesNet-specific
    args.enc_in = enc_in
    args.dec_in = enc_in
    # checkpoint was trained to predict single target channel
    args.c_out = 1
    args.top_k = 2
    args.num_kernels = 6
    args.dropout = 0.1

    return args


def postprocess_and_save(setting, out_dir='./evaluation_results'):
    res_folder = os.path.join('./results', setting)
    if not os.path.exists(res_folder):
        raise SystemExit(f'Result folder not found: {res_folder}')

    pred_path = os.path.join(res_folder, 'pred.npy')
    true_path = os.path.join(res_folder, 'true.npy')
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise SystemExit('pred.npy or true.npy missing in results folder')

    preds = np.load(pred_path)
    trues = np.load(true_path)

    # reshape: (N, L, D) -> flatten along samples and horizon
    N, L, D = preds.shape
    preds_flat = preds.reshape(-1, D)
    trues_flat = trues.reshape(-1, D)

    # we'll save CSV for the first target dimension (assume target is last column)
    col_idx = -1
    pred_1d = preds_flat[:, col_idx]
    true_1d = trues_flat[:, col_idx]

    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame({'true': true_1d, 'pred': pred_1d})
    df_out.to_csv(os.path.join(out_dir, 'predictions_vs_true_pl1.csv'), index=False)

    # metrics
    errors = pred_1d - true_1d
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = sqrt(mse)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.nanmean(np.abs(errors / true_1d)) * 100
    denom = (np.abs(pred_1d) + np.abs(true_1d))
    smape = np.nanmean(np.where(denom==0, 0, 2 * np.abs(pred_1d - true_1d) / denom)) * 100

    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE(%)': mape, 'sMAPE(%)': smape}

    # plots: set Times New Roman
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']

    # true vs pred
    plt.figure(figsize=(12,5))
    plt.plot(true_1d, label='True')
    plt.plot(pred_1d, label='Pred')
    plt.xlabel('Sample index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('True vs Pred', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'true_vs_pred_pl1.png'))
    plt.close()

    # error over time
    plt.figure(figsize=(12,4))
    plt.plot(errors, label='Error (pred - true)')
    plt.xlabel('Sample index', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Prediction Error over Time', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_over_time_pl1.png'))
    plt.close()

    # print metrics
    print('Metrics:')
    for k,v in metrics.items():
        print(f'  {k}: {v}')

    # also save metrics file
    with open(os.path.join(out_dir, 'metrics_pl1.txt'), 'w') as f:
        for k,v in metrics.items():
            f.write(f'{k}: {v}\n')

    print(f'Wrote CSV and plots to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False,
                        default='/data/sda/CYZ/Time-Series-Library/checkpoints/long_term_forecast_custom_timesnet_TimesNet_custom_ftMS_sl12_ll6_pl1_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth')
    parser.add_argument('--data_path', type=str, required=False, default='./datasets/adjusted_data.csv')
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        raise SystemExit(f'Checkpoint not found: {ckpt}')

    setting = os.path.basename(os.path.dirname(ckpt))
    print('Using setting:', setting)

    eval_args = make_args(setting, args.data_path, seq_len=12, label_len=6, pred_len=1)

    # instantiate experiment which builds the model
    exp = Exp_Long_Term_Forecast(eval_args)

    # run test (will load checkpoint and apply inverse if args.inverse==True)
    exp.test(setting, test=1)

    # postprocess saved results
    postprocess_and_save(setting)


if __name__ == '__main__':
    main()
