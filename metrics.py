#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 9:04
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : metrics.py
# @Software: PyCharm

from tqdm import tqdm
import os
import numpy as np
import torch


def calc_metrics(model, feat_path, lbl_dict, L, device, thres=0.5):
    """
    Calculate J_A,J_S,J_E,J_B, VACC according to {Evaluating VAD for Automatic Speech Recognition, ICSP2014, Sibo Tong et al.}
    :param L: The adjust length in that paper
    :param thres: threshold. Frames with predicion higher than thres will be determined to be speech.
    :return: ACC(J_A), J_S(SBA), J_E(EBA), J_B(BP), VACC
    """
    all_pred = []
    all_lbls = []
    for audio_file in tqdm(os.listdir(feat_path)):
        audio_id = audio_file.split('.')[0]
        frames = np.load(os.path.join(feat_path, audio_file))

        inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)

        pred = torch.squeeze(model(inp))
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)
    all_pred = np.array(all_pred)
    all_lbls = np.array(all_lbls)
    all_pred[all_pred >= thres] = 1
    all_pred[all_pred < thres] = 0
    acc = (all_pred + all_lbls) % 2
    acc = (len(all_pred) - acc.sum()) / len(all_pred)

    R, M = 0, 0  # R for speech segments in label, M for from VAD
    Js = []
    Je = []
    for i in tqdm(range(len(all_pred) - 1)):
        if all_pred[i] == 0 and all_pred[i + 1] == 1:
            M += 1
        if all_lbls[i] == 0 and all_lbls[i + 1] == 1:
            R += 1  # speech segment begins
            begin_seg_lbl = all_lbls[i + 1:i + L + 1]
            begin_seg_pred = all_pred[i + 1:i + L + 1]
            js = ((begin_seg_pred + begin_seg_lbl) % 2).sum()
            Js.append((L - js) / L)
        if all_lbls[i] == 1 and all_lbls[i + 1] == 0:
            end_lbl = all_lbls[i - L + 1:i + 1]
            end_pred = all_pred[i - L + 1:i + 1]
            je = ((end_lbl + end_pred) % 2).sum()
            Je.append((L - je) / L)
    Js = sum(Js) / R
    Je = sum(Je) / R
    Jb = R * (Js + Je) / (2 * M)
    vacc = 4 / (1 / acc + 1 / Js + 1 / Je + 1 / Jb)

    return acc, Js, Je, Jb, vacc
