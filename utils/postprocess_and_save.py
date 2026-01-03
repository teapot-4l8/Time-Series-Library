import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
import sys
from sklearn.metrics import r2_score

def postprocess_and_save(setting, data_path, out_dir='./evaluation_results/group_6'):
    res_folder = os.path.join('./results', setting)
    if not os.path.exists(res_folder):
        raise SystemExit(f'Result folder not found: {res_folder}')

    preds = np.load(os.path.join(res_folder, 'pred.npy'))   # (N, L, D)
    trues = np.load(os.path.join(res_folder, 'true.npy'))

    N, L, D = preds.shape
    preds_flat = preds.reshape(-1, D)
    trues_flat = trues.reshape(-1, D)

    # 变量名顺序（与 Dataset_Custom 完全一致）
    df = pd.read_csv(data_path)
    var_names = list(df.columns)
    if 'date' in var_names:
        var_names.remove('date')

    os.makedirs(out_dir, exist_ok=True)

    # 字体设置：Times New Roman
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']

    metrics_rows = []

    for var in TARGET_VARS:
        if var not in var_names:
            raise ValueError(f'Variable {var} not found in data')

        idx = var_names.index(var)

        pred_1d = preds_flat[:, idx]
        true_1d = trues_flat[:, idx]

        err = pred_1d - true_1d

        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))

        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.nanmean(np.abs(err / true_1d)) * 100

        denom = np.abs(pred_1d) + np.abs(true_1d)
        smape = np.nanmean(
            np.where(denom == 0, 0, 2 * np.abs(err) / denom)
        ) * 100

        r2 = r2_score(true_1d, pred_1d)

        metrics_rows.append({
            'Variable': var,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE(%)': mape,
            'sMAPE(%)': smape,
            'R2': r2
        })

        # ========= 保存 CSV =========
        df_out = pd.DataFrame({
            'true': true_1d,
            'pred': pred_1d
        })
        df_out.to_csv(
            os.path.join(out_dir, f'pred_vs_true_{var}.csv'),
            index=False
        )

        # ========= 画 True vs Pred 折线图 =========
        plt.figure(figsize=(12, 5))
        plt.plot(true_1d, label='True', linewidth=1.5)
        plt.plot(pred_1d, label='Pred', linewidth=1.5)
        plt.xlabel('Sample index', fontsize=12)
        plt.ylabel(var, fontsize=12)
        plt.title(f'True vs Pred — {var}', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f'true_vs_pred_{var}.png'),
            dpi=300
        )
        plt.close()

    # ========= 汇总指标 =========
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(
        os.path.join(out_dir, 'metrics_per_variable.csv'),
        index=False
    )

    print('\nEvaluation Metrics (per variable):')
    print(df_metrics.to_string(index=False))
    print(f'\nAll results saved to: {out_dir}')
    
    

