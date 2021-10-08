#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 17:24
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : run.py
# @Software: PyCharm
import argparse
import random
from datetime import datetime

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

'''
深度学习训练的整体过程
'''


def run():
    parser = argparse.ArgumentParser(description='train VAD')
    parser.add_argument('--h5_file_path', default='./data/h5df/train_with_noises_feat.h5', type=str)
    parser.add_argument('--h5_label_path', default='./data/h5df/labels_with_noises.h5', type=str)
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
    parser.add_argument('--log_path', default='./log/', type=str)

    args = parser.parse_args()
    today = datetime.today()
    time_str = today.strftime("%Y-%m-%d-%H:%M:%S")
    log_path = os.path.join(args.log_path, f'train_log_{time_str}.log')
    logger.add(log_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1.加载数据集
    train_loader, val_loader = datasets.getdataloader(args.h5_file_path, args.h5_label_path)
    vad_net = RNN(args.input_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, bidirectional=True,
                  device=device).to(
        device)

    # 2.定义代价函数和优化器模块
    # Binary Cross Entropy
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(vad_net.parameters(), lr=args.lr)
    report_interval = args.report_interval
    interval_loss = 0

    # 3.训练过程
    logger.info("start train vad_net on {}".format(device))
    losses = []
    aucs = []
    eers = []
    fprs = []
    tprs = []
    for epoch in range(args.num_epoch):
        vad_net.train()
        for i, data in enumerate(train_loader, 0):
            x, y = data
            logger.debug("x.type : {}, y.type : {}".format(type(x), type(y)))
            logger.debug("epoch = {}, i = {}, x.shape = {}, y.shape = {}".format(epoch, i, x.shape, y.shape))
            inp = x.float().to(device)
            target = y.float().to(device)
            loss = train_gru(vad_net, inp, target, criterion, optimizer, device=device)
            if i % report_interval == 0:
                logger.info(
                    "epoch = {}, batch n = {}, average loss = {}".format(epoch, i, interval_loss / report_interval))
                losses.append(interval_loss / report_interval)
                interval_loss = 0
            else:
                interval_loss += loss
        vad_net.eval()

        # 在验证集上进行验证
        logger.info("============start running in val datasets===============")
        auc, eer, fpr, tpr = evaluate(vad_net, val_loader, device=device)
        logger.info("epoch: {}, (auc, eer) = {:.5f}, {:.5f}".format(epoch, auc, eer))
        logger.info("============end running in val datasets===============")
        aucs.append(auc)
        eers.append(eer)
        fprs.append(fpr)
        tprs.append(tpr)
        #
        # plt.plot(fpr, tpr, '-.', linewidth=3,
        #          label="epoch {}\nAUC={:.4f},EER={:.4f}".format(epoch, auc, eer))
        torch.save(vad_net.state_dict(), os.path.join(args.checkpoint_path, f"epoch_{epoch}.pth"))
    # 画出各种性能指标的图 loss，auc，err

    # 4.评估
    # 在测试集上进行测试，并进一步评估性能，画出各种结果图
    # TODO:考虑需要哪些图
    # 先把测试集构造出来，再做进一步的考虑

    # 最终在vad_src_test.wav上进行结果展示


if __name__ == '__main__':
    run()
