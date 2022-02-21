# encoding: utf-8
# author: zhaotianhong
# contact: zhaoteanhong@gmail.com

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from dataloader import load_matrix, load_data, load_feas, data_transform_with_fea
from model import GDLGF, GDLBE_Conv, GDLBE_EB, GDLBE
from sklearn.preprocessing import StandardScaler
from utils import scaled_laplacian, cheb_poly, evaluate_model, evaluate_metric

'''
参数部分
'''
'''
参数部分
'''


# torch.backends.cudnn.benchmark=True

def arg_set(model_name, version, timesteps, pre_n, fea_c=[1, 9, 8], Ks=3, Kt=3):
    '''
    配置参数
    :param model_name: 模型名称
    :param version: 训练版本
    :param timesteps: 历史观测时间
    :param pre_n:预测未来的时间点
    :param fea_c:特征通道[历史观测，建成环境特征，全局特征]
    :param Ks:空间卷积核
    :param Kt:时间卷积核
    :return:
    '''
    ap = argparse.ArgumentParser(description="my setting")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ap.add_argument("--device", default=device)
    ap.add_argument("--version", type=str, default=version)  # 保存的版本名
    ap.add_argument("--path_matrix", type=str, default="dataset/A_1139.csv")
    ap.add_argument("--path_x_data", type=str, default="dataset/V_1139.csv")
    ap.add_argument("--dir_global", type=str, default="dataset/global")

    ap.add_argument("--day_slot", type=int, default=288)  # 一天的观测值数量
    ap.add_argument("--tvt", type=list, default=[11, 3, 7])  # 训练、验证和测试比例,天数
    ap.add_argument("--fea_channel", type=list, default=fea_c)  # 数据通道维度
    ap.add_argument("--n_timesteps", type=int, default=timesteps)  # 观测节点数
    ap.add_argument("--n_pred", type=int, default=pre_n)  # 预测节点数（未来第三个节点，也就是说15分钟以后的量）
    ap.add_argument("--nodes", type=int, default=1139)  # 节点数量

    ap.add_argument("--Ks", type=int, default=Ks)  # 空卷积核大小
    ap.add_argument("--Kt", type=int, default=Kt)  # 时卷积核大小
    ap.add_argument("--lr", type=float, default=1e-3)  # 学习率
    ap.add_argument("--drop_prob", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)

    if model_name == "GDLGF":
        ap.add_argument("--dir_fea", type=str, default="dataset/sbe")
        ap.add_argument("--model_name", type=str, default="GDLGF")
    elif model_name == "GDLBE_Conv":
        ap.add_argument("--dir_fea", type=str, default="dataset/sbe")
        ap.add_argument("--model_name", type=str, default="GDLBE_Conv")
    elif model_name == "GDLBE_EB":
        ap.add_argument("--dir_fea", type=str, default="dataset/sbe_int")
        ap.add_argument("--model_name", type=str, default="GDLBE_EB")
    elif model_name == "GDLBE":
        ap.add_argument("--dir_fea", type=str, default="dataset/dbe")
        ap.add_argument("--model_name", type=str, default="GDLBE")

    args = ap.parse_args()
    return args


def init_dir(args):
    '''
    初始化log文件夹
    :param args:
    :return:
    '''
    dir_version = os.path.join("save", "version_%s" % (args.version))
    if not os.path.exists(dir_version):
        os.makedirs(dir_version)
    dir_save = os.path.join(dir_version, "%s_his_%s_pre_%s" % (args.model_name, args.n_timesteps, args.n_pred))
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    return dir_save


def get_lap_kernel(args):
    '''
    计算图卷积核函数
    :param path_matrix:
    :param Ks:
    :return:
    '''
    W = load_matrix(args.path_matrix)  # 原始邻接矩阵
    L = scaled_laplacian(W)  # 拉普普拉斯矩阵
    Lk = cheb_poly(L, args.Ks)  # 契比雪夫矩阵
    Lk = torch.Tensor(Lk.astype(np.float32)).to(args.device)  # 变为tensor
    return Lk


def dataset(args):
    '''
    加载数据集
    :param args:
    :return:
    '''
    train_feas, val_feas, test_feas = load_feas(args.dir_global, args.dir_fea,
                                                args.tvt[0] * args.day_slot, args.tvt[1] * args.day_slot)
    train, val, test = load_data(args.path_x_data, args.tvt[0] * args.day_slot,
                                 args.tvt[1] * args.day_slot)  # 加载预测数据，并切分训练数据集

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)  # 归一化处理

    # 数据处理为tensor
    x_train, y_train = data_transform_with_fea(train, train_feas, args.n_timesteps, args.n_pred, args.fea_channel,
                                               args.day_slot, args.device)
    x_val, y_val = data_transform_with_fea(val, val_feas, args.n_timesteps, args.n_pred, args.fea_channel,
                                           args.day_slot, args.device)
    x_test, y_test = data_transform_with_fea(test, test_feas, args.n_timesteps, args.n_pred, args.fea_channel,
                                             args.day_slot, args.device)

    # 形成数据集
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, args.batch_size)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(test_data, args.batch_size)

    return train_iter, val_iter, test_iter, scaler


def modeling(args, Lk):
    '''
    初始化模型
    :param args:
    :param Lk:
    :return:
    '''
    # 定义模型：损失函数，模型结构，优化器
    loss = nn.MSELoss()
    if args.model_name == "GDLGF":
        model = GDLGF(args.n_timesteps, args.nodes, args.fea_channel, args.drop_prob, Lk, args.Kt, args.Ks).to(
            args.device)
    elif args.model_name == "GDLBE_Conv":
        model = GDLBE_Conv(args.n_timesteps, args.nodes, args.fea_channel, args.drop_prob, Lk, args.Kt, args.Ks).to(
            args.device)
    elif args.model_name == "GDLBE_EB":
        model = GDLBE_EB(args.n_timesteps, args.nodes, args.fea_channel, args.drop_prob, Lk, args.Kt, args.Ks).to(
            args.device)
    elif args.model_name == "GDLBE":
        model = GDLBE(args.n_timesteps, args.nodes, args.fea_channel, args.drop_prob, Lk, args.Kt, args.Ks).to(
            args.device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)  # weight_decay=0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # weight_decay=0.001
    schedu_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    model_info = get_parameter_number(model)

    return model, loss, optimizer, schedu_lr, args.model_name + model_info


def get_parameter_number(net):
    '''
    模型参数统计
    :param net:
    :return:
    '''
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total parameter:', total_num, 'Trainable parameter:', trainable_num)
    return "Total parameter:%s Trainable parameter:%s" % (total_num, trainable_num)


def train(model_name, version, timesteps, pre_n):
    '''
    训练
    :param model_name: 模型名称：[GDLGF, GDLBE_Conv, GDLBE_EB, GDLBE]
    :param version: 版本
    :param timesteps: 历史观测步长
    :param pre_n: 预测步长
    :return:
    '''
    args = arg_set(model_name, version, timesteps, pre_n)
    Lk = get_lap_kernel(args)
    model, loss, optimizer, schedu_lr, model_info = modeling(args, Lk)
    train_iter, val_iter, test_iter, scaler = dataset(args)
    dir_save = init_dir(args)

    min_val_loss = np.inf
    start_time = time.time()
    loss_path = os.path.join(dir_save, "log.txt")

    for epoch in range(1, args.epochs + 1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        schedu_lr.step()
        val_loss = evaluate_model(model, loss, val_iter)
        if val_loss < min_val_loss:
            # best_epoch = epoch
            min_val_loss = val_loss
            save_path = os.path.join(dir_save, "epoch_%s.pt" % (epoch))
            torch.save(model.state_dict(), save_path)
        loss_path = os.path.join(dir_save, "log.txt")
        with open(loss_path, 'a+') as fw:
            fw.write("%s,%s,%s\n" % (epoch, l_sum / n, val_loss))
        print("epoch", epoch, ", Train loss:", l_sum / n, ", Validation loss:", val_loss)

    train_time = time.time() - start_time
    with open(loss_path, 'a+') as fw:
        fw.write(model_info + "Train time:%s\n" % (train_time))
        fw.write("***** Train Done *****\n")


def interface(model_name, version, timesteps, pre_n, show=False):
    '''
    测试
    :param model_name: 模型名称
    :param version: 版本
    :param timesteps: 历史观测步长
    :param pre_n: 预测步长
    :param show: 是否可视化
    :return:
    '''
    args = arg_set(model_name, version, timesteps, pre_n)
    Lk = get_lap_kernel(args)
    model, loss, optimizer, schedu_lr, model_info = modeling(args, Lk)
    train_iter, val_iter, test_iter, scaler = dataset(args)
    dir_save = init_dir(args)

    epochs = []
    for name in os.listdir(dir_save):
        if "pt" in name:
            epochs.append(int(name.split("_")[1][:-3]))
    epoch = max(epochs)

    model_path = os.path.join(dir_save, "epoch_%s.pt" % (epoch))
    print("test: ", model_path)
    loss_path = os.path.join(dir_save, "log.txt")

    model.load_state_dict(torch.load(model_path))
    l = evaluate_model(model, loss, test_iter)
    MAE, RMSE = evaluate_metric(model, test_iter, scaler, show, model_name)

    with open(loss_path, 'a+') as fw:
        fw.write("\n***** Test *****\n")
        fw.write("%s,%s,%s\n" % ("LOSS", "MAE", "RMSE"))
        fw.write("%s,%s,%s\n" % (l, MAE, RMSE))

    print("Test loss:", l, "MAE:", MAE, ", RMSE:", RMSE)

    return (args.version, args.model_name, timesteps, pre_n, l, MAE, RMSE)


if __name__ == '__main__':
    version = 'V1'
    model_name = "GDLBE"
    timesteps, pre_n =  12, 3

    train(model_name, version, timesteps, pre_n)
    interface(model_name, version, timesteps, pre_n, show=False)

