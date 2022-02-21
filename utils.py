# encoding: utf-8
# author: zhaotianhong
# contact: zhaoteanhong@gmail.com


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def scaled_laplacian(A):
    '''
    生成拉普拉斯算子
    :param A:
    :return:
    '''
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    '''
    生成切比雪夫多项式
    :param L:
    :param Ks:
    :return:
    '''
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def evaluate_model(model, loss, data_iter):
    '''模型测试'''
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            # 是一个min_batch评估，【minbatch,1,timestep,nodes_len】
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler, show, model_name):
    '''
    评估模型
    :param model:模型
    :param data_iter:测试数据
    :param scaler:数据分布
    :param show:是否可视化
    :return:
    '''
    model.eval()
    y_show, y_pred_show = [], []
    with torch.no_grad():
        mae, mse = [], []
        for x, y in data_iter:
            # 为可视化准备数据
            y_show.extend(scaler.inverse_transform(y.cpu().numpy()))
            y_p = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy())
            y_p[y_p < 0] = 0
            y_p = (y_p + 0.5).astype(int)
            y_pred_show.extend(y_p)
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            y_pred[y_pred < 0] = 0
            y_pred = (y_pred + 0.5).astype(int)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            # mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        # MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())

        # 可视化数据
        y_show_df, y_pred_show_df = pd.DataFrame(y_show), pd.DataFrame(y_pred_show)
        if show:
            for i in range(5):
                y = list(y_show_df[i])
                y_pred = list(y_pred_show_df[i])
                plot_result(y, y_pred, model_name)
                if i == 5:
                    break
        return MAE, RMSE


def plot_result(y, y_pred, model_name):
    '''
    可视化查看预测结果
    :param y:
    :param y_pred:
    :return:
    '''
    plt.figure(figsize=(6, 3.5))
    plt.plot(y, label='Truth')
    plt.plot(y_pred, 'r--', label=model_name)
    plt.ylabel("Passenger demand", size=14)
    plt.xlabel("Time interval", size=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
