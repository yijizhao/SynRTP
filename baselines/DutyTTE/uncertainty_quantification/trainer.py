import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from utils.argparser import ws
import time
from tqdm import tqdm
import os

class Dataset_list(Dataset):
    def __init__(self, xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od):
        self.xs = xs
        self.segment_travel_time_mean = segment_travel_time_mean
        self.total_ts = total_ts
        self.segment_travel_time = segment_travel_time
        self.segment_num = segment_num
        self.ts_10min = ts_10min
        self.od = od
        
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.segment_travel_time_mean[idx], self.total_ts[idx], self.segment_travel_time[idx], self.segment_num[idx], self.ts_10min[idx],  self.od[idx]

def collate_fn_list(batch):
    xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = zip( *batch)

    xs = torch.tensor(xs).long()
    segment_travel_time_mean = torch.tensor(segment_travel_time_mean, dtype=torch.float32)
    total_ts = torch.tensor(total_ts, dtype=torch.float32)
    segment_travel_time = torch.tensor(segment_travel_time, dtype=torch.float32)
    segment_num = torch.tensor(segment_num).long()
    ts_10min = torch.tensor(ts_10min).long()
    od = torch.tensor(od).long()

    return xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)


class CustomDataset(Dataset):
    def __init__(self, xs, nodes, segment_travel_time_mean, start_timestamp, start_ts, total_ts, segment_travel_time,
                 segment_num, ts_10min, od, start_day):
        self.xs = xs
        self.nodes = nodes
        self.start_timestamp = start_timestamp
        self.start_ts = start_ts
        self.total_ts = total_ts
        self.segment_travel_time = segment_travel_time
        self.segment_num = segment_num
        self.ts_10min = ts_10min
        self.od = od
        self.start_day = start_day
        self.segment_travel_time_mean = segment_travel_time_mean

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        sample = {
            'xs': self.xs[idx],
            'nodes': self.nodes[idx],
            'start_timestamp': self.start_timestamp[idx],
            'segment_travel_time_mean': self.segment_travel_time_mean[idx],
            'start_ts': self.start_ts[idx],
            'total_ts': self.total_ts[idx],
            'segment_travel_time': self.segment_travel_time[idx],
            'segment_num': self.segment_num[idx],
            'ts_10min': self.ts_10min[idx],
            'od': self.od[idx],
            'start_day': self.start_day[idx]
        }
        return sample


import pickle


def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws = get_workspace()

class Trainer:
    def __init__(self, model: nn.Module, device, args):
        self.model = model
        self.device = device
        self.rho = args.rho
        self.early_stop = args.early_stop

    def generated_path_eta_uq(self, args):
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)
        optimizer = torch.optim.Adam(self.model.parameters(), args.lr)

        pickle_file_path_train = os.path.join(
            ws, f"datasets/dutytte/{args.dataset_name}/train.pkl"
        )
        if os.path.exists(pickle_file_path_train):
            with open(pickle_file_path_train, 'rb') as f:
                data_to_load_train = pickle.load(f)
        print('loaded train data: ', pickle_file_path_train)

        pickle_file_path_val = os.path.join(
            ws, f"datasets/dutytte/{args.dataset_name}/val.pkl"
        )
        if os.path.exists(pickle_file_path_val):
            with open(pickle_file_path_val, 'rb') as f:
                data_to_load_val = pickle.load(f)
        print('loaded val data: ', pickle_file_path_val)

        pickle_file_path_test = os.path.join(
            ws, f"datasets/dutytte/{args.dataset_name}/test.pkl"
        )

        if os.path.exists(pickle_file_path_test):
            with open(pickle_file_path_test, 'rb') as f:
                data_to_load_test = pickle.load(f)
        print('loaded test data: ', pickle_file_path_test)

        def loss_fn(y_pred_mean, bias_lower, bias_upper, y_true):
            y_pred_upper =  y_pred_mean + bias_upper 
            y_pred_lower = y_pred_mean - bias_lower 

            rho = self.rho 
            loss0 = torch.abs(y_pred_mean - y_true)
            loss1 = torch.max(y_true - y_pred_upper, torch.tensor([0.]).cuda(1)) * 2 / rho
            loss2 = torch.max(y_pred_lower - y_true, torch.tensor([0.]).cuda(1)) * 2 / rho
            loss3 = torch.abs(y_pred_upper - y_pred_lower)
            loss = loss0  + loss1 + loss2  + loss3
            return loss.mean() 

        def mis(y_pred_mean, bias_lower, bias_upper, y_true):
            y_pred_upper = y_pred_mean + bias_upper
            y_pred_lower = y_pred_mean - bias_lower
            rho = self.rho 
            loss0 = np.abs(y_pred_mean - y_true) 

            loss1 = np.max(y_true - y_pred_upper, 0) * 2 / rho

            loss2 = np.max(y_pred_lower - y_true, 0) * 2 / rho #

            loss3 = np.abs(y_pred_upper - y_pred_lower)
            loss = loss0 + loss1 + loss2 + loss3
            return loss.mean() 

        def picp(y_pred_mean, bias_lower, bias_upper, y_true):
            y_pred_upper = y_pred_mean + bias_upper
            y_pred_lower = y_pred_mean - bias_lower
            picp = (((y_true < y_pred_upper.reshape(-1)) & (y_true > y_pred_lower.reshape(-1))) + 0).sum() / len(y_true)
            return picp 

        traindataset = Dataset_list(data_to_load_train['train_xs'],
                          data_to_load_train['segment_travel_time_mean_train'], 
                          data_to_load_train['total_ts_train'],
                          data_to_load_train['segment_travel_time_train'],
                          data_to_load_train['segment_num_train'], 
                          data_to_load_train['ts_10min_train'],
                          data_to_load_train['od_train']) 

        traindataloader = DataLoader(traindataset, batch_size=64, shuffle=False, collate_fn=collate_fn_list, drop_last=True) # 128

        valdataset = Dataset_list(data_to_load_val['val_xs'],
                          data_to_load_val['segment_travel_time_mean_val'], 
                          data_to_load_val['total_ts_val'],
                          data_to_load_val['segment_travel_time_val'],
                          data_to_load_val['segment_num_val'], 
                          data_to_load_val['ts_10min_val'],
                          data_to_load_val['od_val']) 

        valdataloader = DataLoader(valdataset, batch_size=64, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        testdataset = Dataset_list(data_to_load_test['test_xs'],
                          data_to_load_test['segment_travel_time_mean_test'], 
                          data_to_load_test['total_ts_test'],
                          data_to_load_test['segment_travel_time_test'],
                          data_to_load_test['segment_num_test'], 
                          data_to_load_test['ts_10min_test'],
                          data_to_load_test['od_test'])

        testdataloader = DataLoader(testdataset, batch_size=64, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        train_loss = []
        early_stop = EarlyStop(mode='minimize', patience=self.early_stop)
        result_file = f'val_.{args.dataset_name}.txt'  

        for epoch in range(args.n_epoch):
            if early_stop.stop_flag: break

            self.model.train()
            print('train epoch {}'.format(epoch))
            for batch in tqdm(traindataloader):
                xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                predict_mean, bias_lower, bias_upper, load_balancing_loss = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                mis_loss = loss_fn(predict_mean.reshape(-1), bias_lower.reshape(-1), bias_upper.reshape(-1), total_ts.reshape(-1).float().to(self.device))
                if args.load_balancing:
                    mis_loss += load_balancing_loss * args.load_balancing_weight
                optimizer.zero_grad()
                mis_loss.backward()
                optimizer.step()
                train_loss.append(mis_loss.item())
            print(f'training... loss of epoch: {epoch}: ' + str((sum(train_loss) / len(train_loss))))

            if epoch % 1 == 0:
                print(f'validation... of epoch {epoch}')
                predicts = []
                predicts_bias_lower = []
                predicts_bias_upper = []
                label = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(valdataloader):
                        xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                        predict_mean, bias_lower, bias_upper, load_balancing_loss = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                        total_ts = pad_sequence(total_ts, batch_first=True, padding_value=0).float()
                        predicts += predict_mean.reshape(-1).tolist()
                        predicts_bias_lower += bias_lower.reshape(-1).tolist()
                        predicts_bias_upper += bias_upper.reshape(-1).tolist()
                        label += total_ts.reshape(-1).float().tolist()

                    predicts = np.array(predicts).reshape(-1)
                    label = np.array(label).reshape(-1)
                    predicts_bias_lower = np.array(predicts_bias_lower).reshape(-1)
                    predicts_bias_upper = np.array(predicts_bias_upper).reshape(-1)
                    from sklearn.metrics import mean_squared_error as mse
                    from sklearn.metrics import mean_absolute_error as mae
                    def mape_(label, predicts):
                        return (abs(predicts - label) / label).mean()

                    def acc_eta_global(label, predicts, top_n):
                        absolute_errors = np.abs(predicts - label)
                        count_within_tolerance = np.sum(absolute_errors <= top_n)
                        return count_within_tolerance / len(label)

                    val_mape = mape_(label, predicts)
                    val_mse = mse(label, predicts)
                    val_mae = mae(label, predicts)

                    acc_eta_list = [10, 20, 30, 40, 50, 60]

                    val_acc_eta_values = [] 
                    for n_val in acc_eta_list:
                        acc_eta_n = acc_eta_global(label, predicts, n_val)
                        val_acc_eta_values.append(acc_eta_n) 

                    predicts_upper = predicts + predicts_bias_upper
                    predicts_lower = predicts - predicts_bias_lower
                    val_width = predicts_upper - predicts_lower
                    val_mis = mis(predicts, predicts_bias_lower, predicts_bias_upper, label)
                    val_picp = picp(predicts, predicts_bias_lower, predicts_bias_upper, label)

                    print('val point estimation: MAPE:%.3f\tRMSE:%.2f\tMAE:%.2f' % (val_mape * 100, np.sqrt(val_mse), val_mae))
                    result = (
                        f'mae:{val_mae:.3f} | rmse:{np.sqrt(val_mse):.3f} | mape:{val_mape * 100:.3f} | '
                        f'acc_eta@10:{val_acc_eta_values[0]:.3f} | acc_eta@20:{val_acc_eta_values[1]:.3f} | acc_eta@30:{val_acc_eta_values[2]:.3f}'
                    )
                is_best_change = early_stop.append(val_mae)
                with open(result_file, 'a') as file:
                    file.write(f'Epoch {epoch}: {result} | Best epoch: {early_stop.best_epoch} \n')

                if is_best_change:
                    dir_check(ws + f"/model_params/MoEUQ/{local_time}/")
                    best_model_path = ws + f"/model_params/MoEUQ/{local_time}/finished_{epoch}.pth"
                    torch.save(self.model.state_dict(), best_model_path)
                    print('val best model saved at: ', best_model_path)
                    self.model.load_state_dict(torch.load(best_model_path))
                    print('val best model loaded')
        print("testing...")
        print('load best val model at: ', best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))
        print('testing, best val model loaded')

        predicts = []
        predicts_bias_lower = []
        predicts_bias_upper = []
        label = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(testdataloader):
                xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                predict_mean, bias_lower, bias_upper, load_balancing_loss = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                total_ts = pad_sequence(total_ts, batch_first=True, padding_value=0).float()

                predicts += predict_mean.reshape(-1).tolist()
                predicts_bias_lower += bias_lower.reshape(-1).tolist()
                predicts_bias_upper += bias_upper.reshape(-1).tolist()
                label += total_ts.reshape(-1).float().tolist()

            predicts = np.array(predicts).reshape(-1)
            label = np.array(label).reshape(-1)
            predicts_bias_lower = np.array(predicts_bias_lower).reshape(-1)
            predicts_bias_upper = np.array(predicts_bias_upper).reshape(-1)
            from sklearn.metrics import mean_squared_error as mse
            from sklearn.metrics import mean_absolute_error as mae
            def mape_(label, predicts):
                return (abs(predicts - label) / label).mean()

            def acc_eta_global(label, predicts, top_n):
                absolute_errors = np.abs(predicts - label)               
                count_within_tolerance = np.sum(absolute_errors <= top_n)
                return count_within_tolerance / len(label)

            test_mape = mape_(label, predicts)
            test_mse = mse(label, predicts)
            test_mae = mae(label, predicts)

            acc_eta_list = [10, 20, 30, 40, 50, 60]

            test_acc_eta_values = [] 
            for n_val in acc_eta_list:
                acc_eta_n = acc_eta_global(label, predicts, n_val)
                test_acc_eta_values.append(acc_eta_n) 

            predicts_upper = predicts + predicts_bias_upper
            predicts_lower = predicts - predicts_bias_lower
            test_width = predicts_upper - predicts_lower
            test_mis = mis(predicts, predicts_bias_lower, predicts_bias_upper, label)
            test_picp = picp(predicts, predicts_bias_lower, predicts_bias_upper, label)

            print('test point estimation: MAPE:%.3f\tRMSE:%.2f\tMAE:%.2f' % (test_mape * 100, np.sqrt(test_mse), test_mae))
            result = (
                        f'mae:{test_mae:.3f} | rmse:{np.sqrt(test_mse):.3f} | mape:{test_mape * 100:.3f} | '
                        f'acc_eta@10:{test_acc_eta_values[0]:.3f} | acc_eta@20:{test_acc_eta_values[1]:.3f} | acc_eta@30:{test_acc_eta_values[2]:.3f}'
                    )
            with open(result_file, 'a') as file:
                file.write(f'Evaluation in test: {result} | Best epoch: {early_stop.best_epoch} \n')


class EarlyStop():

    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def whether_stop(metric_lst=[], n=2, mode='maximize'):
    if len(metric_lst) < 1: return False  # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n
