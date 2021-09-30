#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 14:38
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : datasets.py
# @Software: PyCharm
import torch
from torch.utils.data import Dataset, DataLoader
from h5py import File

'''
数据集的构建：
    数据集提取的特征以h5df格式存放，同时其标签也以h5df格式存放。
    将训练集和验证集放在一起，之后再随机选取。
    
'''


class AudioTrainDataSet(Dataset):
    def __int__(self, h5FilePath, h5LabelPath, transform=None):
        super(AudioTrainDataSet, self).__init__()
        self._datas = File(h5FilePath, 'r')
        self._labels = File(h5LabelPath, 'r')
        self._len = len(self._labels)
        self._transform = transform

        self.idx_to_item = {
            idx: item
            for idx, item in enumerate(self._labels.keys())
        }

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        wav_id = self.idx_to_item[idx]
        data = self._datas[wav_id]
        labels = self._labels[wav_id]

        data = torch.as_tensor(data).float()
        labels = torch.as_tensor(labels).float()
        return data, labels


def getdataloader(h5filepath, h5labelpath, transform=None, **kwargs):
    data = AudioTrainDataSet(h5filepath, h5labelpath, transform)
    return DataLoader(data, 2, shuffle=True, **kwargs)
