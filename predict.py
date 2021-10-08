#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 15:00
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : predict.py
# @Software: PyCharm
import argparse
import json

import librosa
import torch
import numpy as np
import os
from model.models import RNN
from visual import visual
from data.extract_feature import extract_feature_for_test
from loguru import logger

# 作为一个全局变量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pred_to_vad_label(
        prediction,
        frame_size: float = 0.032,
        frame_shift: float = 0.008,
        threshold: float = 0.5,
):
    """Convert model prediction to VAD labels.

    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extracting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extracting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following example for more detailed explanation.

    Examples:
        >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >>> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = {"speech_times": []}

    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames["speech_times"].append({"start_time": frame2time(start), "end_time": frame2time(end)})
            # speech_frames.append(
            #     "{:.3f},{:.3f}".format(frame2time(start), frame2time(end))
            # )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames["speech_times"].append({"start_time": frame2time(start), "end_time": frame2time(end)})
            # speech_frames.append(
            #     "{:.3f},{:.3f}".format(frame2time(start), frame2time(end))
            # )
        prev_state = state
    return speech_frames


def predict(model, wav_file, device, win_len=0.032, win_hop=0.008):
    model.eval()
    frames = extract_feature_for_test(wav_file, sr=8000, win_len=win_len, win_hop=win_hop, input_dim=14)
    inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)
    pred = torch.squeeze(model(inp))
    speech_frames = pred_to_vad_label(pred, win_len, win_hop, 0.5)
    return speech_frames


def main():
    parser = argparse.ArgumentParser(
        description='extract_feature for vad train')
    parser.add_argument('--fs', default=8000, type=int)
    parser.add_argument('--win_len', default=0.032, type=float)
    parser.add_argument('--win_hop', default=0.008, type=float)
    parser.add_argument('--input_dim', default=14, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--save_path', default="./checkpoints", type=str)
    parser.add_argument('--epoch_num', default=9, type=int)
    parser.add_argument('--test_file', default='vad_test_src.wav', type=str)

    args = parser.parse_args()

    # 获取测试文件
    root = os.getcwd()
    data_path = os.path.join(root, 'data', 'test_wav')
    test_file = os.path.join(data_path, args.test_file)

    # 加载模型及其参数
    vad_net = RNN(args.input_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, bidirectional=True,
                  device=device).to(
        device)
    vad_net.load_state_dict(torch.load(os.path.join(args.save_path, f"epoch_{args.epoch_num}.pth")))

    # 预测
    pred_labels = predict(vad_net, test_file, device=device)
    logger.debug(pred_labels)

    # 可视化
    wav_array, sr = librosa.load(test_file, sr=args.fs)
    visual(sr=args.fs, signal=wav_array, labels=pred_labels, signal_id='vad_src_wav')


if __name__ == '__main__':
    main()
