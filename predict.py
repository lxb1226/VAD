#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 15:00
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : predict.py
# @Software: PyCharm

import torch
import numpy as np
import os
from utils.vad_utils import prediction_to_vad_label
from model.models import RNN


def predict(model, feat_file, device, win_len=0.032, win_hop=0.008):
    model.eval()
    frames = np.load(feat_file)
    inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)
    pred = torch.squeeze(model(inp))

    line = prediction_to_vad_label(pred, win_len, win_hop, 0.5)

    with open('./data/test_wav/result.txt', 'w') as f:
        f.write(line)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = 14  # 1(energy)+13(mfcc)
    hidden_size = 128
    num_layers = 1
    save_path = "./checkpoints"
    # vad_net = VADnet(input_dim, hidden_size=hidden_size, num_layers=num_layers).to(device)
    vad_net = RNN(input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, device=device).to(
        device)
    vad_net.load_state_dict(torch.load(os.path.join(save_path, f"epoch_9.pth")))
    feat_file = "./data/test_wav/vad_test_src.npy"
    predict(vad_net, feat_file, device=device)


if __name__ == '__main__':
    main()
