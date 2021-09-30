#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 17:24
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : run.py
# @Software: PyCharm
import argparse
import random
from tqdm import tqdm
import numpy as np
import os
from loguru import logger
import matplotlib.pyplot as plt

from train import train_gru
from eval import evaluate

import datasets
import torch
import torch.nn as nn

from model.models import RNN

# 训练整体过程
# TODO: 构建整体的流程
'''
    1.目前数据集并没有构建完整，但是整体的格式以及代码已经完整，后续的构建比较简单。
    2.先构建好整体运行的函数，跑通再说；最好明天构建完，国庆就可以跑通，然后就可以构建更多的数据集以及选择不同的模型来进行比较。
    3.对于copy过来的代码还需要进一步看看有没有错误；
    4.后续如何评估还需要进一步的考虑。
'''


def run():
    parser = argparse.ArgumentParser(description='train VAD')
    parser.add_argument('--h5_file_path', default='./data/h5df/train_feat.h5', type=str)
    parser.add_argument('--h5_label_path', default='./data/h5df/labels.h5', type=str)
    parser.add_argument('--fs', default=8000, type=int)
    parser.add_argument('--win_len', default=0.032, type=float)
    parser.add_argument('--win_hop', default=0.008, type=float)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--report_interval', default=50, type=int)
    parser.add_argument('--input_dim', default=14, type=int)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', type=str)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1.加载数据集
    train_loader = datasets.getdataloader(args.h5filepath, args.h5labelpath)
    vad_net = RNN(args.input_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, bidirectional=True,
                  device=device).to(
        device)

    # 2.定义代价函数和优化器模块
    # Binary Cross Entropy
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(vad_net.parameters(), lr=args.lr)
    report_interval = args.report_interval
    interval_loss = 0

    # 3.构建训练过程
    for epoch in range(args.num_epoch):
        vad_net.train()
        for i, data in enumerate(train_loader, 0):
            x, y = data
            inp = torch.from_numpy(x).float().to(device)
            target = torch.from_numpy(y).float().to(device)
            loss = train_gru(vad_net, inp, target, criterion, optimizer)
            if i % report_interval == 0:
                logger.info(
                    "epoch = {}, batch n = {}, average loss = {}".format(epoch, i, interval_loss / report_interval))
                interval_loss = 0
            else:
                interval_loss += loss
        vad_net.eval()

        auc, eer, fpr, tpr = evaluate(vad_net, dev_feat_path, dev_lbl_dict)

        plt.plot(fpr, tpr, '-.', linewidth=3,
                 label="epoch {}\nAUC={:.4f},EER={:.4f}".format(epoch, auc, eer))

        print('======> epoch {}, (auc, eer)={:.5f},{:.5f}'.format(epoch, auc, eer))
        torch.save(vad_net.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pth"))
    # 4.评估


if __name__ == '__main__':
    run()
