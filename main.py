#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 10:10
# @Author  : heyjude
# @Email   : 1944303766@qq.com
# @File    : main.py
# @Software: PyCharm

import argparse
import json
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import spafe.features.mfcc as mfcc
import spafe.utils.preprocessing as preprocess
import torch
import torch.nn as nn
from tqdm import tqdm

from eval import get_metrics
from model.models import DnnVAD, RNN, VADnet
from utils.vad_utils import parse_vad_label, prediction_to_vad_label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def vis_sample(model, lbl_dict, feat_path, fig_path, save=True):
    """
    Visualize a sample prediction and its label
    :param model: the VAD model
    :param lbl_dict: map of sample ID and its label
    :param feat_path: sample feature path
    :param fig_path: where to store the figure
    :param save: if save, then save the figure to fig_path
    """
    sample_path = os.listdir(feat_path)[np.random.randint(len(os.listdir(feat_path)))]
    sample_id = sample_path.split('.')[0]

    sample = np.load(os.path.join(feat_path, sample_path))
    sample_label = lbl_dict[sample_id]

    hiddens = model.init_hiddens(1)
    hiddens = (hiddens[0].to(device), hiddens[1].to(device))

    sample_pred = model(torch.from_numpy(sample).unsqueeze(1).float().to(device))
    sample_pred = sample_pred.squeeze()
    sample_pred = sample_pred.detach().cpu().numpy()
    auc_nosmooth, eer_nosmooth, _, _ = get_metrics(sample_pred, sample_label)

    plt.figure(figsize=(8, 4))
    plt.plot(sample_label, c='r', label='sample label')
    plt.plot(sample_pred + 0.01, c='g', linestyle='-.',
             label='prediction\n auc,eer={:.4f}, {:.4f}'.format(auc_nosmooth, eer_nosmooth))
    plt.legend(fontsize=10)
    plt.xlabel('frame', fontsize=12)
    if save:
        plt.savefig(os.path.join(fig_path, '{}_pred.png'.format(sample_id)), dpi=120)
    plt.show()


def train_dnn(dnn_net, inp, target, criterion, optimizer):
    # if inp.ndim == 2:
    #     inp.unsqueeze_(1).float()
    # if target.ndim == 1:
    #     target.unsqueeze_(1).float()
    inp.to(device)
    target.to(device)

    optimizer.zero_grad()
    outputs = dnn_net(inp)
    preds = torch.argmax(outputs, dim=1)
    # preds = preds.float()
    loss = criterion(outputs, target.long())
    loss.backward()
    optimizer.step()
    return loss.item()


def train_rnn(gru_net, inp, target, criterion, optimizer):
    if inp.ndim == 2:
        inp.unsqueeze_(1).float()
    if target.ndim == 1:
        target.unsqueeze_(1).float()
    inp.to(device)
    target.to(device)

    optimizer.zero_grad()
    pred = torch.squeeze(vad_net(inp))
    loss = criterion(pred, target.squeeze_())
    loss.backward()
    optimizer.step()
    return loss.item()


def train_lstm(vad_net, inp, target, criterion, optimizer):
    """
    :param inp: (seq_len, batch_size, feat_dim) seq_len * batch_size(1) *feat_dim
    :param vad_net: the model
    :param target: (seq_len, batch_size)
    :return: prediction (0-1) and loss (float)
    """
    # print(id(vad_net))
    if inp.ndim == 2:
        inp.unsqueeze_(1).float()
    if target.ndim == 1:
        target.unsqueeze_(1).float()
    inp.to(device)
    target.to(device)

    hiddens = vad_net.init_hiddens(batch_size=1)
    hiddens = (hiddens[0].to(device), hiddens[1].to(device))
    optimizer.zero_grad()
    pred = torch.squeeze(vad_net(inp, hiddens))
    loss = criterion(pred, target.squeeze_())
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_dnn(model, feat_path, lbl_dict):
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

        # inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)
        inp = torch.from_numpy(frames).float().to(device)

        outputs = model(inp)
        pred = torch.argmax(outputs, dim=1)
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)

    auc, eer, fpr, tpr = get_metrics(all_pred, all_lbls)
    return auc, eer, fpr, tpr


def evaluate_rnn(model, feat_path, lbl_dict):
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

        hiddens = model.init_hiddens(batch_size=1)
        hiddens = (hiddens[0].to(device), hiddens[1].to(device))

        pred = torch.squeeze(model(inp))
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)

    auc, eer, fpr, tpr = get_metrics(all_pred, all_lbls)
    return auc, eer, fpr, tpr

def evaluate_lstm(model, feat_path, lbl_dict):
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

        hiddens = model.init_hiddens(batch_size=1)
        hiddens = (hiddens[0].to(device), hiddens[1].to(device))

        pred = torch.squeeze(model(inp, hiddens))
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)

    auc, eer, fpr, tpr = get_metrics(all_pred, all_lbls)
    return auc, eer, fpr, tpr


def calc_metrics(model, feat_path, lbl_dict, L, thres=0.5):
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


def predict(args, model, feat_path, target_path):
    """
    :param model: vad model
    :param feat_path: features path of dev data
    :param target_path: where test_label_task2.txt is going to be stored
    """
    model.eval()
    with open(os.path.join(target_path, 'test_label_task2.txt'), 'w') as f:
        for testfile in tqdm(os.listdir(feat_path)):
            test_id = testfile.split('.')[0]
            frames = np.load(os.path.join(feat_path, testfile))
            inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)

            # hiddens = model.init_hiddens(batch_size=1)
            # hiddens = (hiddens[0].to(device), hiddens[1].to(device))

            pred = torch.squeeze(model(inp))

            line = prediction_to_vad_label(pred, args.win_len, args.win_hop, 0.5)
            f.write(test_id + " " + line + '\n')


def extract_feature(audio_path, label, sr=8000, win_len=0.032, win_hop=0.008):
    audio_data, _ = librosa.load(audio_path, sr=sr)
    audio_framed, frame_len = preprocess.framing(audio_data, fs=sr, win_len=win_len,
                                                 win_hop=win_hop)
    frame_num = audio_framed.shape[0]
    # assert frame_num >= len(label), "frame_num : {}, len of labels : {}".format(frame_num, len(label))
    if frame_num > len(label):
        label += [0] * (frame_num - len(label))
    else:
        label = label[: frame_num]
    frame_energy = (audio_framed ** 2).sum(1)[:, np.newaxis]
    frame_mfcc = mfcc.mfcc(audio_data, fs=sr, win_len=win_len, win_hop=win_hop)
    # 联结帧能量以及mfcc特征
    frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
    # 将特征 + 能量 保存到文件中
    return label, frame_feats


def process_data(data_list, feat_path, json_path):
    lbl_dict = {}
    with open(data_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            audio_path = data[0].strip()
            label = ' '.join(data[1:])
            # logger.debug('label : {}'.format(label))
            label = parse_vad_label(label)
            label, frame_feats = extract_feature(audio_path, label)
            audio_id = audio_path.strip().split('\\')[-1].split('.')[0]
            # logger.debug("audio_id : {}, label : {}".format(audio_id, label))
            np.save(os.path.join(feat_path, audio_id + '.npy'), frame_feats)
            lbl_dict[audio_id] = label
    # 保存到json文件中
    json_str = json.dumps(lbl_dict)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run LSTM for VAD')
    parser.add_argument('--fs', default=8000, type=int)
    parser.add_argument('--win_len', default=0.032, type=float)
    parser.add_argument('--win_hop', default=0.008, type=float)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--report_interval', default=50, type=int)
    parser.add_argument('--stage', default=1, type=int)
    parser.add_argument('--L', default=5, type=int)  # adjust length in VACC calculation
    parser.add_argument('--model_type', default='lstm', type=str)
    parser.add_argument('--data_path', default=r'F:\workspace\GHT\projects\vad\data', type=str, help='data path')
    parser.add_argument('--data_list', default=r'F:\workspace\GHT\projects\vad\data\labels\train_labels.txt')
    parser.add_argument('--val_list', default=r'F:\workspace\GHT\projects\vad\data\labels\val_labels.txt')

    args = parser.parse_args()

    # 一些文件夹的配置选项
    root = os.getcwd()
    data_path = args.data_path
    train_path = os.path.join(data_path, "dataset", "train")
    val_path = os.path.join(data_path, "dataset", "val")
    feat_path = os.path.join(data_path, "feat")
    train_feat_path = os.path.join(feat_path, "train")

    val_feat_path = os.path.join(feat_path, "val")
    labels_path = os.path.join(data_path, 'labels')

    fig_path = os.path.join(root, "figs")
    save_path = os.path.join(root, "checkpoints")

    train_labels_path = os.path.join(labels_path, r'train_lbl_dict.json')
    val_labels_path = os.path.join(labels_path, r'val_lbl_dict.json')

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
    for path in [train_feat_path, val_feat_path, fig_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    if args.stage <= 0:
        # 数据预处理
        # TODO：考虑将数据预处理以及加载封装成一个类
        print('stage 0: data preparation and feature extraction')
        # 加载训练数据集
        process_data(args.data_list, train_feat_path, train_labels_path)
        # 加载测试数据集
        process_data(args.val_list, val_feat_path, val_labels_path)

    with open(train_labels_path, 'r') as f:
        train_lbl_dict = json.load(f)
    with open(val_labels_path, 'r') as f:
        val_lbl_dict = json.load(f)
    # 构建模型

    input_dim = 14  # 1(energy)+13(mfcc)
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    # 选择训练模型所需要的参数以及函数
    model_type = args.model_type
    if model_type == 'dnn':
        vad_net = DnnVAD(input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        evaluate = evaluate_dnn
        train = train_dnn
    elif model_type == 'rnn':
        vad_net = RNN(input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, device=device).to(
            device)
        criterion = nn.BCELoss()
        evaluate = evaluate_rnn
        train = train_rnn
    elif model_type == 'lstm':
        vad_net = VADnet(input_dim, hidden_size=hidden_size, num_layers=num_layers).to(device)
        criterion = nn.BCELoss()
        evaluate = evaluate_lstm
        train = train_lstm

    optimizer = torch.optim.Adam(vad_net.parameters(), lr=args.lr)
    report_interval = args.report_interval
    interval_loss = 0

    if args.stage <= 1:
        # 模型训练过程
        # TODO：封装成一个函数
        print('stage 1: model training')
        # here we first implement the case where batch_size = 1
        plt.figure()
        plt.xlabel('FPR', fontsize=13)
        plt.ylabel("TPR", fontsize=13)
        plt.title('RoC curve', fontsize=15)
        plt.plot(np.linspace(0, 1, 1000), np.linspace(1, 0, 1000), '.', linewidth=0.5, markersize=3, color='cyan')

        for epoch in range(args.num_epoch):
            vad_net.train()
            for i, audio_file in tqdm(
                    enumerate(random.sample(os.listdir(train_feat_path), len(os.listdir(train_feat_path))))):
                audio_id = audio_file.split('.')[0]
                audio = np.load(os.path.join(train_feat_path, audio_file))
                # frame_num * input_dim
                inp = torch.from_numpy(audio).float().to(device)
                audio_path = train_path + "\\" + audio_id + ".wav"
                # logger.debug("audio_path : {}".format(audio_path))
                target = torch.tensor(train_lbl_dict[audio_id]).float().to(device)
                # loss = trainGru(vad_net, inp, target, criterion, optimizer)
                # loss = train_rnn(vad_net, inp, target, criterion, optimizer)
                # loss = train_dnn(vad_net, inp, target, criterion, optimizer)
                loss = train(vad_net, inp, target, criterion, optimizer)
                if i % report_interval == 0:
                    print("epoch = ", epoch, "batch n = ", i, " average loss = ", interval_loss / report_interval)
                    interval_loss = 0
                else:
                    interval_loss += loss
            vad_net.eval()
            # auc, eer, fpr, tpr = evaluate_dnn(vad_net, val_feat_path, val_lbl_dict)
            auc, eer, fpr, tpr = evaluate(vad_net, val_feat_path, val_lbl_dict)

            plt.plot(fpr, tpr, '-.', linewidth=3,
                     label="epoch {}\nAUC={:.4f},EER={:.4f}".format(epoch, auc, eer))

            print('======> epoch {}, (auc, eer)={:.5f},{:.5f}'.format(epoch, auc, eer))
            torch.save(vad_net.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pth"))
        plt.legend(fontsize=14, loc="lower center")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(os.path.join(fig_path, 'roc_curve.png'), dpi=100)

    # if args.stage <= 2:
    #     # 模型评估
    #     # TODO：考虑封装成一个函数
    #     print('stage 2: model evaluating')
    #
    #     vad_net.load_state_dict(torch.load(os.path.join(save_path, f"epoch_{args.num_epoch - 1}.pth")))
    #
    #     # vis_sample(vad_net, test_lbl_dict, test_feat_path, fig_path, save=True)
    #     # vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)
    #     acc, Js, Je, Jb, vacc = calc_metrics(vad_net, test_feat_path, test_lbl_dict, L=args.L, thres=0.5)
    #     print('>=== ACC = {:.4f}'.format(acc))
    #     print('>=== SBA = {:.4f}'.format(Js))
    #     print('>=== EBA = {:.4f}'.format(Je))
    #     print('>=== BP = {:.4f}'.format(Jb))
    #     print('>=== VADD = {:.4f}'.format(vacc))

    # if args.stage <= 3:
    #     # 模型在测试集的评估 即最终结果
    #     # TODO：同样考虑封装成一个函数
    #     print('stage 3: model predicting')
    #     vad_net.load_state_dict(torch.load(os.path.join(save_path, f"epoch_{args.num_epoch - 1}.pth")))
    #     acc, Js, Je, Jb, vacc = calc_metrics(vad_net, test_feat_path, test_lbl_dict, L=args.L, thres=0.5)
    # vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)
    # vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)

    print('DONE!')
