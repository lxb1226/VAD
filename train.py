#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 9:04
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : train.py
# @Software: PyCharm
import torch


def train_gru(vad_net, inp, target, criterion, optimizer, device):
    inp.to(device)
    target.to(device)
    optimizer.zero_grad()
    pred = torch.squeeze(vad_net(inp))
    loss = criterion(pred, target.squeeze_())
    loss.backward()
    optimizer.step()
    return loss.item()
