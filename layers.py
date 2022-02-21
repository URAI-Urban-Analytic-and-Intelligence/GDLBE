# encoding: utf-8
# author: zhaotianhong
# contact: zhaoteanhong@gmail.com


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            out = F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
            return out
        return x


class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.c_in = c_in
        self.c_out = c_out
        self.conv_c = nn.Conv2d(c_in, c_out, (kt, 1))
        self.conv_t = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)

    def forward(self, x):
        if self.c_in < self.c_in:
            x_c = F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])[:, :, self.kt - 1:, :]
        else:
            x_c = self.conv_c(x)
        x_t = self.conv_t(x)
        out = (x_t[:, :self.c_out, :, :] + x_c) * torch.sigmoid(x_t[:, self.c_out:, :, :])
        return out


class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


class Hadamard_product_connect(nn.Module):
    def __init__(self, nodes):
        super(Hadamard_product_connect, self).__init__()

        self.p1 = Parameter(torch.randn(nodes, nodes))
        self.p2 = Parameter(torch.randn(nodes, nodes))
        # self.bn = nn.BatchNorm1d(nodes)
        # self.p3 = Parameter(torch.Tensor(nodes, nodes))

    def forward(self, v1, v2):
        v1 = torch.einsum('ikj,kk->ikj', [v1, self.p1])
        v2 = torch.einsum('ikj,kk->ikj', [v2, self.p2])
        # v3 = torch.einsum('ijk,kk->ijk', [v3, self.p3])

        out = torch.add(v1, v2)
        # out_bn = self.bn(out)
        return out


class LSTM_layer(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(LSTM_layer, self).__init__()
        hidden_size = 32
        # self.bn = nn.BatchNorm2d(input_channel)
        self.rnn = nn.LSTM(input_size=input_channel, hidden_size=hidden_size, num_layers=1, dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channel)
        self.conv = nn.Conv2d(out_channel, out_channel,(1,1))

    def forward(self, x):
        """
        :param X: Input data of shape (batch_size, channel, timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_nodes)
        """
        b, c, t, n = x.shape
        # x = self.bn(x)
        x_v = x.permute(0, 3, 2, 1).contiguous().view(-1, t, c)  # in: (batch, time_step, nodes)
        r_out, hc = self.rnn(x_v)
        out_fc = self.fc(r_out)
        lstm_x = out_fc.view(b, n, t, -1).permute((0, 3, 2, 1))
        out = torch.relu(self.conv(lstm_x)) + lstm_x
        return out


class Global_fea_embed(nn.Module):
    def __init__(self, input_channel, out_channel, in_feas_len, out_feas_len):
        super(Global_fea_embed, self).__init__()
        self.bn = nn.BatchNorm2d(input_channel)
        self.conv = nn.Conv2d(input_channel, out_channel, (1, 1))
        self.out = nn.Linear(in_feas_len, out_feas_len)

    def forward(self, x):
        x_bn = self.bn(x)
        x_conv = self.conv(x_bn).permute((0, 1, 3, 2))
        out = self.out(x_conv).permute((0, 1, 3, 2))
        out = torch.sigmoid(out)
        return out


class Output_time_conv(nn.Module):
    def __init__(self, c, timestep, n):
        super(Output_time_conv, self).__init__()
        self.tconv1 = temporal_conv_layer(timestep, c, c)  # Kt,  c_in, c_out
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1,1))
        self.fc = nn.Conv2d(c, 1, (1,1))

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = torch.sigmoid(self.tconv2(x_ln) + x_ln)
        out = self.fc(x_t2)
        return out


class GCRNN_Unit(nn.Module):
    def __init__(self, kt, ks, in_c, hide_c, out_c, dropout, nodes, Lk):
        super(GCRNN_Unit, self).__init__()
        self.t_conv = temporal_conv_layer(kt, in_c, hide_c)
        self.s_conv = spatio_conv_layer(ks, hide_c, Lk)
        self.lstm = LSTM_layer(hide_c, out_c, dropout)
        self.ln = nn.LayerNorm([nodes, out_c])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_t1 = self.t_conv(x)
        x_s = self.s_conv(x_t1)
        x_t2 = self.lstm(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class GDL_Module(nn.Module):
    def __init__(self, kt, ks, cs_iho, dropout, nodes, Lk):
        super(GDL_Module, self).__init__()
        c_iho_1 = cs_iho[0]
        c_iho_2 = cs_iho[1]
        self.gcrnn_1 = GCRNN_Unit(kt, ks, c_iho_1[0], c_iho_1[1], c_iho_1[2], dropout, nodes, Lk)
        self.gcrnn_2 = GCRNN_Unit(kt, ks, c_iho_2[0], c_iho_2[1], c_iho_2[2], dropout, nodes, Lk)

    def forward(self, x):
        st_x1 = self.gcrnn_1(x)
        st_x2 = self.gcrnn_2(st_x1)
        return st_x2
