#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 15:20
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : visual.py
# @Software: PyCharm

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from loguru import logger


def plot_signal(sr, signal, labels, signal_id):
    logger.info(
        f"Sampling rate = {sr} | Num. points = {len(signal)} | Tot. duration = {len(signal) / sr:.2f} s"
    )
    plt.figure(figsize=(15, 10))
    sns.set()
    sns.lineplot(x=[i / sr for i in range(len(signal))], y=signal)

    start, end = 0, 0
    for seg in labels["speech_segments"]:
        plt.axvspan(end, seg["start_time"], alpha=0.5, color="r")  # 噪声段
        start, end = seg["start_time"], seg["end_time"]
        plt.axvspan(start, end, alpha=0.5, color="g")  # 语音段
    plt.axvspan(end, (len(signal) - 1) / sr, alpha=0.5, color="r")

    plt.title(f"Sample number {signal_id} with speech in green", size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="visualize raw data")
    parser.add_argument(
        "--data-dir", type=str, default=r"E:\workspace\GHT\code\VAD\data\test_wav"
    )
    args = parser.parse_args()
    data_path = os.path.join(args.data_dir, "vad_test_src.wav")
    label_path = os.path.join(args.data_dir, "result.txt")
    signal, sr = sf.read(data_path)

    # 读取标签
    with open(label_path, 'r') as f:
        line = f.read()
    labels = {}
    labels['speech_segments'] = []
    for item in line.split(" "):
        res = item.split(',')
        labels['speech_segments'].append({'start_time': float(res[0]), 'end_time': float(res[1])})
    logger.info("labels : {}".format(labels['speech_segments']))
    # Plot
    logger.info("Plotting signal ...")
    plot_signal(sr, signal, labels, 0)


if __name__ == "__main__":
    main()
