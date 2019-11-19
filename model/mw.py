'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-18 10:23:32
@LastEditors: fangn
@LastEditTime: 2019-11-19 10:23:12
'''
from model import common
import torch
import torch.nn as nn
import scipy.io as sio


# 创建模型
def make_model(args, parent=False):
    return BSR(args)


def matrix_init():
    a = torch.linspace(1, 15, steps=15) - 8
    a = a.float()

    mat_1 = torch.mul(a, a)
    mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
    a = a.view(15, 1)
    mat_2 = torch.mul(a, a.t())
    mat_1 = mat_1.contiguous().view(1, 1, 225, 1, 1)
    mat_2 = mat_2.contiguous().view(1, 1, 225, 1, 1)
    mat_3 = mat_3.contiguous().view(1, 1, 225, 1, 1)

    return torch.cat((mat_1, mat_2, mat_3), 1).cuda()


# BSR 模型
class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(BSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3  # 卷积核大小
        self.scale_idx = 0

        act = nn.ReLU(True)  # 激活函数

        self.DWT = common.DWT()  # 二维离散小波
        self.IWT = common.IWT()  # 逆向的二维离散小波

        n = 3
        # downsample的第一层，维度变化4->16
        m_head = [common.BBlock(conv, 4, 160, 3, act=act)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(common.BBlock(conv, 160, 160, 3, act=act))

        # downsample的第二层，维度变化640->256（默认的feature map == 64）
        d_l2 = [common.BBlock(conv, 640, n_feats * 4, 3, act=act)]
        for _ in range(n):
            d_l2.append(
                common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

        # downsample的第三层，并与upsample进行连接，也是upsample的第三层
        # 维度变化1024->256，256->1024
        pro_l3 = [common.BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act)]
        for _ in range(n * 2):
            pro_l3.append(
                common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        pro_l3.append(
            common.BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act))

        # upsample的第二层，维度变化256->640
        i_l2 = []
        for _ in range(n):
            i_l2.append(
                common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        i_l2.append(common.BBlock(conv, n_feats * 4, 640, 3, act=act))

        # upsample的第一层，维度变化160->4
        i_l1 = []
        for _ in range(n):
            i_l1.append((common.BBlock(conv, 160, 160, 3, act=act)))
        m_tail = [conv(160, 4, 3)]

        # downsample的第一层
        self.head = nn.Sequential(*m_head)
        self.d_l1 = nn.Sequential(*d_l1)
        # downsample的第二层
        self.d_l2 = nn.Sequential(*d_l2)
        # 第三层连接层
        self.pro_l3 = nn.Sequential(*pro_l3)
        # upsample的第二层
        self.i_l2 = nn.Sequential(*i_l2)
        # upsample的第一层
        self.i_l1 = nn.Sequential(*i_l1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # downsample的第一层
        x1 = self.d_l1(self.head(self.DWT(x)))
        # downsample的第二层
        x2 = self.d_l2(self.DWT(x1))
        # upsample的第三层，并且使用了short cut的结构，将对应的downsample的第二层
        # 加到upsample对应的层上。
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        # upsample的第二层
        x_ = self.IWT(self.i_l2(x_)) + x1
        # upsample的第一层
        x = self.IWT(self.tail(self.i_l1(x_))) + x
        # x = self.add_mean(x)
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
