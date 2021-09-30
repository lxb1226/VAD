#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 16:47
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : data_visual.py
# @Software: PyCharm
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from loguru import logger


def data_visual(sr, signal, labels, signal_id):
    logger.info(
        f"Sampling rate = {sr} | Num. points = {len(signal)} | Tot. duration = {len(signal) / sr:.2f} s"
    )
    plt.figure(figsize=(15, 10))
    sns.set()
    sns.lineplot(x=[i / sr for i in range(len(signal))], y=signal)

    start, end = 0, 0
    for seg in labels["speech_segments"]:
        plt.axvspan(end, seg["start_time"] / sr, alpha=0.5, color="r")
        start, end = seg["start_time"] / sr, seg["end_time"] / sr
        plt.axvspan(start, end, alpha=0.5, color="g")
    plt.axvspan(end, (len(signal) - 1) / sr, alpha=0.5, color="r")

    plt.title(f"Sample number {signal_id} with speech in green", size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


# TODO: 数据的可视化 待完成
def convert_to_dict(line):
    pass

# 测试程序
def main():
    # 写一个测试程序
    parser = argparse.ArgumentParser(description="visualize raw data")
    parser.add_argument("--data-dir", type=str, default=r'E:\workspace\GHT\code\VAD\data')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, "datasets")
    label_file = os.path.join(args.data_dir, "train_with_noise_labels.txt")


if __name__ == '__main__':
    main()
