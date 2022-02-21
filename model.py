# encoding: utf-8
# author: zhaotianhong
# contact: zhaoteanhong@gmail.com

import torch
import torch.nn as nn
from layers import Global_fea_embed, Output_time_conv, GDL_Module


class GDLGF(nn.Module):
    '''
    使用全局特征
    '''
    def __init__(self, timesteps, nodes, fea_channel, dropout, Lk, kt=3, ks=3):
        super(GDLGF, self).__init__()

        # 各个类型数据通道占比：历史观测值，建成环境，全局数据
        self.x_cut = [timesteps * fea_channel[0], timesteps * fea_channel[1], fea_channel[2]]
        ch_iho_x = [[fea_channel[0], 64, 128], [128, 64, 16]]  # 通道变化[input, hidden, output]
        out_c = ch_iho_x[-1][-1]  # 最后一层通道数量
        out_timesteps = timesteps - (kt - 1) * len(ch_iho_x)  # 最后一层时间步长 ＊时间卷积次数

        self.gdl_x = GDL_Module(kt, ks, ch_iho_x, dropout, nodes, Lk)
        self.em_global = Global_fea_embed(fea_channel[2], out_c, 1, out_timesteps)

        self.output = Output_time_conv(out_c * 2, out_timesteps, nodes)

    def forward(self, x):
        b, c, f, v = x.shape
        historical_x = x[:, :, :self.x_cut[0], :]  # b, c, ts, v
        global_fea = x[:, :, -self.x_cut[2]:, :].view(b, -1, 1, v)

        gdl_x = self.gdl_x(historical_x)  # OUTPUT: batch_size, channel, num_timesteps, num_nodes
        em_gf = self.em_global(global_fea)
        fusion_fea = torch.cat([gdl_x, em_gf], dim=1)
        out = self.output(fusion_fea)

        return out


class GDLBE_Conv(nn.Module):
    '''
    使用卷积的方法处理静态建成环境特征
    '''
    def __init__(self, timesteps, nodes, fea_channel, dropout, Lk, kt=3, ks=3):
        super(GDLBE_Conv, self).__init__()

        # 各个类型数据通道占比：流数据，建成环境，全局数据
        self.x_cut = [timesteps * fea_channel[0], timesteps * fea_channel[1], fea_channel[2]]
        self.timesteps = timesteps
        ch_iho_x = [[fea_channel[0], 64, 128], [128, 64, 16]]  # 通道变化[input, hidden, output]
        ch_iho_be = [[32, 64, 128], [128, 64, 16]]  # 建成环境是9个通道
        out_c = ch_iho_x[-1][-1]  # 最后一层通道数量
        out_timestep = timesteps - (kt - 1) * len(ch_iho_x)  # 最后一层时间步长 ＊时间卷积次数

        self.Conv_sbe = nn.Conv2d(fea_channel[1], 32, 1)
        self.GDL_x = GDL_Module(kt, ks, ch_iho_x, dropout, nodes, Lk)
        self.GDL_sbe = GDL_Module(kt, ks, ch_iho_be, dropout, nodes, Lk)
        self.EB_global = Global_fea_embed(fea_channel[2], out_c, 1, out_timestep)

        self.output = Output_time_conv(out_c * 3, out_timestep, nodes)

    def forward(self, x):
        b, c, f, v = x.shape
        historical_x = x[:, :, :self.x_cut[0], :]  # b, c, ts, v
        be_fea = x[:, :, self.x_cut[0]:self.x_cut[0] + self.x_cut[1], :].view(b, -1, self.timesteps, v)
        global_fea = x[:, :, -self.x_cut[2]:, :].view(b, -1, 1, v)

        conv_sbe = self.Conv_sbe(be_fea)
        st_x = self.GDL_x(historical_x)  # OUTPUT: batch_size, channel, num_timesteps, num_nodes
        st_be = self.GDL_sbe(conv_sbe)
        em_gf = self.EB_global(global_fea)

        fusion_fea = torch.cat([st_x, st_be, em_gf], dim=1)

        out = self.output(fusion_fea)  # batch_size, num_timesteps, num_nodes

        return out


class GDLBE_EB(nn.Module):
    '''
    使用embedding处理静态建成环境特征
    '''
    def __init__(self, timesteps, nodes, fea_channel, dropout, Lk, kt=3, ks=3):
        super(GDLBE_EB, self).__init__()

        self.x_cut = [timesteps * fea_channel[0], timesteps * fea_channel[1], fea_channel[2]]
        self.timesteps = timesteps
        ch_iho_x = [[fea_channel[0], 64, 128], [128, 64, 16]]  # 通道变化[input, hidden, output]
        ch_iho_be = [[32, 64, 128], [128, 64, 16]]  # 建成环境是9个通道
        out_c = ch_iho_x[-1][-1]  # 最后一层通道数量
        out_timestep = timesteps - (kt - 1) * len(ch_iho_x)  # 最后一层时间步长 ＊时间卷积次数

        self.EB_sbe = nn.Embedding(101, 32)
        self.GDL_x = GDL_Module(kt, ks, ch_iho_x, dropout, nodes, Lk)
        self.GDL_sbe = GDL_Module(kt, ks, ch_iho_be, dropout, nodes, Lk)
        self.EB_global = Global_fea_embed(fea_channel[2], out_c, 1, out_timestep)

        self.output = Output_time_conv(out_c * 3, out_timestep, nodes)

    def forward(self, x):
        b, c, f, v = x.shape
        historical_x = x[:, :, :self.x_cut[0], :]  # b, c, ts, v
        be_fea = x[:, :, self.x_cut[0]:self.x_cut[0] + self.x_cut[1], :].view(b, -1, self.timesteps, v).long()
        global_fea = x[:, :, -self.x_cut[2]:, :].view(b, -1, 1, v)

        em_sbe = self.EB_sbe(be_fea)
        em_sbe_sum = torch.sum(em_sbe, dim=1).permute(0, 3, 1, 2)

        st_x = self.GDL_x(historical_x)  # OUTPUT: batch_size, channel, num_timesteps, num_nodes
        st_be = self.GDL_sbe(em_sbe_sum)
        em_gf = self.EB_global(global_fea)
        fusion_fea = torch.cat([st_x, st_be, em_gf], dim=1)

        out = self.output(fusion_fea)  # batch_size, num_timesteps, num_nodes

        return out


class GDLBE(nn.Module):
    '''
    动态建成环境特征
    '''
    def __init__(self, timesteps, nodes, fea_channel, dropout, Lk, kt=3, ks=3):
        super(GDLBE, self).__init__()

        # 各个类型数据通道占比：流数据，建成环境，全局数据
        self.x_cut = [timesteps * fea_channel[0], timesteps * fea_channel[1], fea_channel[2]]
        self.timesteps = timesteps
        ch_iho_x = [[fea_channel[0], 64, 128], [128, 64, 16]]  # 通道变化[input, hidden, output]
        ch_iho_be = [[32, 64, 128], [128, 64, 16]]  # 建成环境是9个通道
        out_c = ch_iho_x[-1][-1]  # 最后一层通道数量
        out_timestep = timesteps - (kt - 1) * len(ch_iho_x)  # 最后一层时间步长 ＊时间卷积次数

        # self.Conv_x = nn.Conv2d(fea_channel[0], 32, 1)
        self.Conv_sbe = nn.Conv2d(fea_channel[1], 32, 1)
        self.GDL_x = GDL_Module(kt, ks, ch_iho_x, dropout, nodes, Lk)
        self.GDL_sbe = GDL_Module(kt, ks, ch_iho_be, dropout, nodes, Lk)
        self.EB_global = Global_fea_embed(fea_channel[2], out_c, 1, out_timestep)

        self.output = Output_time_conv(out_c * 3, out_timestep, nodes)

    def forward(self, x):
        b, c, f, v = x.shape
        historical_x = x[:, :, :self.x_cut[0], :]  # b, c, ts, v
        be_fea = x[:, :, self.x_cut[0]:self.x_cut[0] + self.x_cut[1], :].view(b, -1, self.timesteps, v)
        global_fea = x[:, :, -self.x_cut[2]:, :].view(b, -1, 1, v)

        # conv_x = self.Conv_x(historical_x)
        conv_sbe = self.Conv_sbe(be_fea)

        st_x = self.GDL_x(historical_x)  # OUTPUT: batch_size, channel, num_timesteps, num_nodes
        st_be = self.GDL_sbe(conv_sbe)
        em_gf = self.EB_global(global_fea)
        fusion_fea = torch.cat([st_x, st_be, em_gf], dim=1)

        out = self.output(fusion_fea)  # batch_size, num_timesteps, num_nodes

        return out
