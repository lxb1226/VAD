#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 16:31
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : models.py
# @Software: PyCharm

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=1, bidirectional=True, device="cpu"):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1  # 双向2 单向1
        self.device = device

        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_size, num_layers=self.num_layers, bias=True, batch_first=False,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=self.num_directions * hidden_size,
                            out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def init_hiddens(self, batch_size):
        # hidden state should be (num_layers*num_directions, batch_size, hidden_size)
        # returns a hidden state and a cell state
        hidden = torch.rand(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return hidden

    def forward(self, input_data):
        '''
        input_data : (seq_len, batchsize, input_dim)
        '''
        batch_size = input_data.size(1)

        hiddens = self.init_hiddens(batch_size).to(self.device)  # 隐藏层h0
        outputs, hiddens = self.gru(input_data, hiddens)
        # outputs: (seq_len, batch_size, num_directions* hidden_size)

        pred = self.fc(outputs)
        pred = self.sigmoid(pred)
        return pred
