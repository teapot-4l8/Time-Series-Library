from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
import json

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        else:
            # for non-M4 datasets, ensure frequency_map exists for loss functions (e.g. MASE)
            if not hasattr(self.args, 'frequency_map'):
                # default to 1 (no seasonal differencing)
                self.args.frequency_map = 1
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()
        # history for recording loss per epoch
        history = {'train': [], 'val': []}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                # Different loss functions have different signatures.
                # Predefined losses (MAPE/MASE/SMAPE) expect (insample, freq, forecast, target, mask).
                # Standard MSELoss expects (prediction, target).
                if self.args.loss == 'MSE':
                    loss_value = criterion(outputs, batch_y)
                else:
                    loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            # record history and write to disk
            history['train'].append(float(train_loss))
            history['val'].append(float(vali_loss))
            try:
                with open(os.path.join(path, 'training_history.json'), 'w') as fh:
                    json.dump(history, fh)
            except Exception:
                pass
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # Load checkpoint to CPU first to avoid issues when checkpoint was saved on different GPU ids
        checkpoint = torch.load(best_model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        # ensure model is on the correct device
        self.model.to(self.device)

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        # Validate over the validation DataLoader to support Dataset_Custom and others.
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_true = batch_y[:, -self.args.pred_len:, f_dim:]
                batch_y_mask = batch_y_mark[:, -self.args.pred_len:, f_dim:]

                if self.args.loss == 'MSE':
                    loss = criterion(outputs, batch_y_true)
                else:
                    # pass insample (use batch_x first channel), frequency_map, forecast, target, mask
                    insample = batch_x.detach().cpu()[:, :, 0]
                    loss = criterion(insample, self.args.frequency_map, outputs.detach().cpu(), batch_y_true.detach().cpu(), batch_y_mask.detach().cpu())
                losses.append(loss.item())

        self.model.train()
        return np.average(losses) if len(losses) > 0 else 0.0

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')

        # Debug: print data file and partition info to confirm which data is used for testing
        try:
            data_fp = os.path.join(self.args.root_path, self.args.data_path)
            if os.path.exists(data_fp):
                df_all = pandas.read_csv(data_fp)
                N = len(df_all)
                num_train = int(N * 0.7)
                num_test = int(N * 0.2)
                num_val = N - num_train - num_test
                border1s = [0, num_train - self.args.seq_len, N - num_test - self.args.seq_len]
                border2s = [num_train, num_train + num_val, N]
                border1 = border1s[2]
                border2 = border2s[2]
                # show a few timestamps to verify
                dates = None
                if 'date' in df_all.columns:
                    try:
                        dates = pandas.to_datetime(df_all['date'])
                    except Exception:
                        dates = pandas.to_datetime(df_all['date'], dayfirst=False, infer_datetime_format=True)
                print(f"Test data file: {data_fp} (rows: {N})")
                print(f"Test partition borders: [{border1}, {border2})  (size: {border2-border1})")
                if dates is not None:
                    print("Test window sample dates:")
                    print(dates.iloc[border1:border1+3].tolist())
                    print("...")
                    print(dates.iloc[max(border1, border2-3):border2].tolist())
        except Exception:
            pass

        if test:
            print('loading model')
            # load checkpoint on CPU first to avoid CUDA device index mismatches
            best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            checkpoint = torch.load(best_model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        preds_list = []
        trues_list = []
        with torch.no_grad():
            # If dataset implements last_insample_window (like M4), keep old behaviour
            if hasattr(train_loader.dataset, 'last_insample_window') and hasattr(test_loader.dataset, 'timeseries'):
                x, _ = train_loader.dataset.last_insample_window()
                y = test_loader.dataset.timeseries
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                x = x.unsqueeze(-1)

                B, _, C = x.shape
                dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
                dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
                outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
                id_list = np.arange(0, B, 1)
                id_list = np.append(id_list, B)
                for i in range(len(id_list) - 1):
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                          dec_inp[id_list[i]:id_list[i + 1]], None)
                    if id_list[i] % 1000 == 0:
                        print(id_list[i])

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()

                preds = outputs
                trues = y
                x = x.detach().cpu().numpy()

                for i in range(0, preds.shape[0], max(1, preds.shape[0] // 10)):
                    gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                    pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            else:
                # Generic evaluation: iterate over test_loader
                for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    outputs = self.model(batch_x, None, dec_inp, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    preds_list.append(outputs.detach().cpu().numpy())
                    trues_list.append(batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())

                if len(preds_list) > 0:
                    preds = np.concatenate(preds_list, axis=0)
                    trues = np.concatenate(trues_list, axis=0)
                else:
                    preds = np.zeros((0, self.args.pred_len, 1))
                    trues = np.zeros((0, self.args.pred_len, 1))

                # flatten last dim if present so trues/preds are (N, pred_len)
                preds_flat = preds.squeeze()
                trues_flat = trues.squeeze()
                if preds_flat.ndim == 1:
                    # when pred_len == 1, squeeze gives (N,), reshape to (N,1)
                    preds_flat = preds_flat.reshape(-1, 1)
                if trues_flat.ndim == 1:
                    trues_flat = trues_flat.reshape(-1, 1)

                # optional visualizations
                for i in range(0, preds_flat.shape[0], max(1, preds_flat.shape[0] // 10)):
                    # use last input window from batch_x for plotting if available
                    last_input = batch_x.detach().cpu().numpy()[i % batch_x.shape[0], :, 0]
                    gt = np.concatenate((last_input, trues_flat[i]), axis=0)
                    pd = np.concatenate((last_input, preds_flat[i]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                

        print('test shape:', preds.shape)

        # Attempt to inverse-transform predictions back to original scale if dataset provides scaler
        try:
            # preds shape: (N, pred_len, C)
            shape = preds.shape
            # use test_loader.dataset's inverse_transform (it fits scaler during __read_data__)
            if hasattr(test_loader.dataset, 'inverse_transform'):
                preds_inv = test_loader.dataset.inverse_transform(preds.reshape(shape[0] * shape[1], -1)).reshape(shape)
                preds_to_save = preds_inv[:, :, 0]
            else:
                preds_to_save = preds[:, :, 0]
        except Exception:
            preds_to_save = preds[:, :, 0]

        forecasts_df = pandas.DataFrame(preds_to_save, columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        # index for custom dataset: simple numeric index
        forecasts_df.index = np.arange(preds_to_save.shape[0])
        forecasts_df.index.name = 'id'
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        return
