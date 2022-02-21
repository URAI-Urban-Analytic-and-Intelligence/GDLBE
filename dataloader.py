# encoding: utf-8
# author: zhaotianhong
# contact: zhaoteanhong@gmail.com


import os
import numpy as np
import pandas as pd
import torch


def load_matrix(file_path):
    '''读取邻接矩阵'''
    return pd.read_csv(file_path, header=None).values.astype(float)


def get_dw_index(pre_index, time_step, time_b):
    index_list = []
    for i in range(1, time_step + 1):
        pre_i = pre_index - (i * time_b)
        if pre_i > 0:
            index_list.append(pre_i)
        else:
            min_i = min(index_list)
            index_list.append(min_i)
    return index_list


def load_data(file_path, len_train, len_val):
    '''读取历史观测数据'''
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test


def load_feas(dir_global, dir_fea, len_train, len_val):
    '''读取建成环境特征数据'''
    be_feas = []
    global_feas = []
    # 　加载建成环境特征
    for name in os.listdir(dir_fea):
        file_path = os.path.join(dir_fea, name)

        # 静态需要归一化
        if "sbe_int" in dir_fea:
            df = pd.read_csv(file_path, header=None).values.astype(int)
            df = df
        else:
            df = pd.read_csv(file_path, header=None).values.astype(float)
            df = (df - df.min()) / (df.max() - df.min())
        list_df = list(df)
        extend_21 = []
        for _ in range(21):
            extend_21.extend(list_df)
        be_feas.append(extend_21)
    feas_be = np.array(be_feas).transpose((1, 0, 2))

    # 全局特征加载
    for name in os.listdir(dir_global):
        file_path = os.path.join(dir_global, name)
        df = pd.read_csv(file_path, header=None).values.astype(float)
        if "Holiday" in name:
            global_feas.append(df)
            continue

        if "W" in name:
            matrix = []
            for values in df:
                lines = []
                for v in values:
                    if v == 0:
                        lines.append([1, 0, 0, 0])
                    if v == 1:
                        lines.append([0, 1, 0, 0])
                    if v == 2:
                        lines.append([0, 0, 1, 0])
                    if v == 3:
                        lines.append([0, 0, 0, 1])
                matrix.append(lines)
            weather_feas = np.array(matrix).transpose((0, 2, 1))
            continue
        else:
            df = (df - df.min()) / (df.max() - df.min())
        global_feas.append(df)
    feas_global = np.array(global_feas).transpose((1, 0, 2))

    all_feas = np.concatenate((feas_be, feas_global, weather_feas), axis=1)
    train_feas = all_feas[: len_train]
    val_feas = all_feas[len_train: len_train + len_val]
    test_feas = all_feas[len_train + len_val:]
    return train_feas, val_feas, test_feas


def data_transform_with_fea(data, feas, n_his, n_pred, channels, day_slot, device):
    '''
    构建feature和target
    '''
    fea_len = channels[0] * n_his + channels[1] * n_his + channels[2]
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, fea_len, n_route])
    y = np.zeros([n_day * n_slot, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            records = data[s:e].reshape(1, channels[0] * n_his, n_route)
            fea_be = feas[s:e, :channels[1], :].reshape(1, channels[1] * n_his, n_route)
            fea_g = feas[e + n_pred - 1, channels[1]:, :].reshape(1, channels[2], n_route)
            records_fea = np.concatenate((records, fea_be, fea_g), axis=1)
            x[t, :, :, :] = records_fea
            y[t] = data[e + n_pred - 1]
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


def data_transform(data, n_his, n_pred, day_slot, device):
    '''构建数据集'''
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    y = np.zeros([n_day * n_slot, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
            y[t] = data[e + n_pred - 1]
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
