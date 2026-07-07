#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.config import cfg
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, roc_auc_score

def prediction(pred_score, true_l):
    # 确保 pred_score 是一维张量
    pred_score = pred_score.view(-1)
    true_l = true_l.view(-1)

    # 使 pred_score 和 true_l 形状匹配
    if pred_score.shape[0] != true_l.shape[0]:
        min_len = min(pred_score.shape[0], true_l.shape[0])
        pred_score = pred_score[:min_len]
        true_l = true_l[:min_len]

    # 🚀 **如果 `true_l` 只有一个类别，则跳过 AUC 计算**
    unique_labels = np.unique(true_l.cpu().numpy())
    if len(unique_labels) == 1:
        print(f"⚠️ Warning: Only one class present in y_true ({unique_labels}). Skipping AUC calculation.")
        return -1, -1, -1, -1, -1  # 返回 -1 表示无法计算 AUC 和 AP

    # 🚀 **稳定性改进：确保 pred_score 是 float**
    pred_binary = (pred_score > 0.5).long().detach().cpu().numpy()
    true_l = true_l.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()

    acc = accuracy_score(true_l, pred_binary)
    ap = average_precision_score(true_l, pred_score)
    f1 = f1_score(true_l, pred_binary, average='macro')
    macro_auc = roc_auc_score(true_l, pred_score, average='macro')
    micro_auc = roc_auc_score(true_l, pred_score, average='micro')

    return acc, ap, f1, macro_auc, micro_auc


def Link_loss_meta(pred, y):
    L = nn.BCELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)
    return loss
