#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 9:03
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : eval.py
# @Software: PyCharm

from sklearn import metrics
from tqdm import tqdm
import os
import torch


def evaluate(model, feat_path, lbl_dict, device):
    """
    :param model: the model to be evaluated
    :param feat_path: the path where validation set features are restored
    :param lbl_dict: a dict, wave id to frame-wise label
    :return: auc, eer, tpr, fpr on the validation set
    """
    all_pred = []
    all_lbls = []
    for audio_file in tqdm(os.listdir(feat_path)):
        audio_id = audio_file.split('.')[0]
        frames = np.load(os.path.join(feat_path, audio_file))

        inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)

        # hiddens = model.init_hiddens(batch_size=1)
        # hiddens = (hiddens[0].to(device), hiddens[1].to(device))

        pred = torch.squeeze(model(inp))
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)

    auc, eer, fpr, tpr = get_metrics(all_pred, all_lbls)
    return auc, eer, fpr, tpr


def compute_eer(target_scores, nontarget_scores):
    """Calculate EER following the same way as in Kaldi.

    Args:
        target_scores (array-like): sequence of scores where the
                                    label is the target class
                                    所有正样本的prediction预测值。
        nontarget_scores (array-like): sequence of scores where the
                                    label is the non-target class
                                    所有负样本的预测值
    Returns:
        eer (float): equal error rate
        threshold (float): the value where the target error rate
                           (the proportion of target_scores below
                           threshold) is equal to the non-target
                           error rate (the proportion of nontarget_scores
                           above threshold)
                           也就是eer对应的阈值8
    """
    assert len(target_scores) != 0 and len(nontarget_scores) != 0
    tgt_scores = sorted(target_scores)
    nontgt_scores = sorted(nontarget_scores)

    target_size = float(len(tgt_scores))
    nontarget_size = len(nontgt_scores)
    target_position = 0
    for target_position, tgt_score in enumerate(tgt_scores[:-1]):
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontgt_scores[nontarget_position] < tgt_score:
            break
    threshold = tgt_scores[target_position]
    eer = target_position / target_size
    return eer, threshold


def get_metrics(prediction, label):
    # 输入一段语音的每帧预测值和帧级标签序列
    """Calculate several metrics for a binary classification task.

    Args:
        prediction (array-like): sequence of probabilities
            e.g. [0.1, 0.4, 0.35, 0.8]
        labels (array-like): sequence of class labels (0 or 1)
            e.g. [0, 0, 1, 1]
    Returns:
        auc: area-under-curve
        eer: equal error rate
    """  # noqa: H405, E261
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # fnr = 1 - tpr
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer, thres = compute_eer(
        [pred for i, pred in enumerate(prediction) if label[i] == 1],
        [pred for i, pred in enumerate(prediction) if label[i] == 0],
    )
    return auc, eer, fpr, tpr


if __name__ == "__main__":
    # 第一个参数为模型预测输出（可以是概率，也可以是二值分类结果）
    # 第二个参数为数据对应的标签
    print(get_metrics([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]))
    # 注意：计算最终指标时，应将整个数据集（而不是在每个样本上单独计算）的所有语音帧预测结果合并在一个list中，
    # 对应的标签也合并在一个list中，然后再调用get_metrics来计算指标
