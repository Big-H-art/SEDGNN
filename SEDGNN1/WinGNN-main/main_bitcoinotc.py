#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import dgl
import math
#import wandb
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from model import WinGNN
from test_new import test
from train_new import train
from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset

import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoinotc', help='Dataset')  # 修改默认数据集为 bitcoinotc
    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device no -1')
    parser.add_argument('--seed', type=int, default=2023, help='split seed')
    parser.add_argument('--repeat', type=int, default=60, help='number of repeat model')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train.')
    parser.add_argument('--out_dim', type=int, default=64, help='model output dimension.')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate.')
    parser.add_argument('--maml_lr', type=float, default=0.0008, help='meta learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--drop_rate', type=float, default=0.16, help='drop meta loss')
    parser.add_argument('--num_layers', type=int, default=2, help='GNN layer num')
    parser.add_argument('--num_hidden', type=int, default=256, help='number of hidden units of MLP')
    parser.add_argument('--dropout', type=float, default=0.1, help='GNN dropout')
    parser.add_argument('--residual', type=bool, default=False, help='skip connection')
    parser.add_argument('--beta', type=float, default=0.89, help='The weight of adaptive learning rate component accumulation')
    parser.add_argument('--weight_param', type=float, default=0.5, help='The weight to balance local and global gradients')
    args = parser.parse_args()

    # 初始化日志
    logger = getLogger(cfg.log_path)

    # 加载数据集
    processed_dir = f"D:/cfx/AGS-main/WinGNN-main/dataset/{args.dataset}/"
    graphs, e_feat, e_time, n_feat = load_r(processed_dir)

    if args.dataset in ["reddit_body", "reddit_title", "as_733",
                          "uci-msg", "bitcoinotc", "bitcoinalpha",
                          'stackoverflow_M']:
        graphs, e_feat, e_time, n_feat = load_r(processed_dir)  # 调用 load_r 函数加载数据
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]
    device = torch.device('cuda' if args.cuda_device >= 0 and torch.cuda.is_available() else 'cpu')

    # 初始化模型训练和测试
    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = None

    for rep in range(args.repeat):
        logger.info(f"===== Starting Experiment Run {rep + 1} =====")
        logger.info(f'num_layers:{args.num_layers}, num_hidden: {args.num_hidden}, lr: {args.lr}, maml_lr:{args.maml_lr}, drop_rate:{args.drop_rate}')

        # 设置随机种子
        torch.manual_seed(args.seed + rep)
        random.seed(args.seed + rep)
        np.random.seed(args.seed + rep)

        # 预处理图数据
        graph_l = []
        for idx, graph in tqdm(enumerate(graphs)):
            graph_d = dgl.from_scipy(graph)
            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])
            graph_d.node_feature = torch.Tensor(n_feat[idx]) if n_feat[idx].shape[0] == n_node else torch.Tensor(graph_l[idx - 1].node_feature)
            graph_d = dgl.add_self_loop(dgl.remove_self_loop(graph_d))
            graph_l.append(graph_d.to(device))

        # 数据集划分
        random.shuffle(graph_l)
        n = math.ceil(len(graph_l) * 0.7)
        train_graphs, test_graphs = graph_l[:n], graph_l[n:]

        # 初始化模型和优化器
        model = WinGNN.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout).to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

        # 训练模型
        best_param = train(args, model, optimizer, device, train_graphs, logger)
        model.load_state_dict(best_param['best_state'])
        S_dw = best_param['best_s_dw']

        # 测试模型
        avg_mrr = test(test_graphs, model, args, logger, S_dw, device)
        if avg_mrr > best_mrr:
            best_model = best_param['best_state']
        all_mrr_avg += avg_mrr

    # 保存最佳模型
    torch.save(best_model, f"model_parameter/{args.dataset}.pkl")
    print(f"Average MRR: {all_mrr_avg / args.repeat}")
