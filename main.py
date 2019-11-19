'''
@Description: 
@Author: fangn
@Github: 
@Date: 2019-11-18 10:23:32
@LastEditors: fangn
@LastEditTime: 2019-11-19 14:02:36
'''
import torch

import utility
import data
import model
import loss

# import h5py
from option import args  # option.py 定义外部参数的获取样式
from trainer import Trainer

torch.set_num_threads(12)  # 设置 CPU 的多核心计算
torch.manual_seed(args.seed)  # 设置随机初始化种子，保证每次的初始化都相同
checkpoint = utility.checkpoint(args)  # 对程序传入的外部参数进行处理

if checkpoint.ok:

    # args.model = 'NL_EST'
    # model1 = model.Model(args, checkpoint)
    #
    # args.model = 'KERNEL_EST'
    # model2 = model.Model(args, checkpoint)
    # args.model = 'BSR'

    # 获取网络模型
    model = model.Model(args, checkpoint)
    # 导入相应的训练或测试数据集
    loader = data.Data(args)
    # 导入 loss function
    loss = loss.Loss(args, checkpoint) if not args.test_only else None

    # 训练或测试模型
    t = Trainer(args, loader, model, loss, checkpoint)

    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
