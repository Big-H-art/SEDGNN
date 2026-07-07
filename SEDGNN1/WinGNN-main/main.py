import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import dgl
import math
# import wandb
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
from dataset_prep import load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uci-msg', help='Dataset')
    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device no -1')
    parser.add_argument('--seed', type=int, default=2023, help='split seed')
    parser.add_argument('--repeat', type=int, default=60, help='number of repeat model')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train.')
    parser.add_argument('--out_dim', type=int, default=64, help='model output dimension.')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate.')
    parser.add_argument('--maml_lr', type=float, default=0.0006, help='meta learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--drop_rate', type=float, default=0.16, help='drop meta loss')
    parser.add_argument('--num_layers', type=int, default=2, help='GNN layer num')
    parser.add_argument('--num_hidden', type=int, default=256, help='number of hidden units of MLP')
    parser.add_argument('--dropout', type=float, default=0.1, help='GNN dropout')
    parser.add_argument('--residual', type=bool, default=False, help='skip connection')
    parser.add_argument('--beta', type=float, default=0.89,
                        help='The weight of adaptive learning rate component accumulation')
    parser.add_argument('--weight_param', type=float, default=0.5,
                        help='The weight to balance local and global gradients')

    args = parser.parse_args()
    logger = getLogger(cfg.log_path)

    # Load dataset
    graphs, e_feat, e_time, n_feat, global_state = load_r(args.dataset)

    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]

    device = torch.device('cpu')

    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = None

    for rep in range(args.repeat):
        logger.info(f"===== Starting Experiment Run {rep + 1} =====")
        logger.info('num_layers:{}, num_hidden: {}, lr: {}, maml_lr:{}, drop_rate:{}, 负样本采样固定'.
                    format(args.num_layers, args.num_hidden, args.lr, args.maml_lr, args.drop_rate))

        torch.manual_seed(args.seed + rep)
        random.seed(args.seed + rep)
        np.random.seed(args.seed + rep)

        graph_l = []
        logger.info(f"===== End of Experiment Run {rep + 1} =====\n")

        # Data processing
        for idx, graph in tqdm(enumerate(graphs)):
            graph_d = dgl.from_scipy(graph)
            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])
            graph_d.node_feature = torch.Tensor(n_feat[idx])

            graph_d = dgl.remove_self_loop(graph_d)
            graph_d = dgl.add_self_loop(graph_d)

            edges = graph_d.edges()
            row = edges[0].numpy()
            col = edges[1].numpy()
            n_e = graph_d.num_edges() - graph_d.num_nodes()

            # 正样本边（真实边）
            pos_src = row[:n_e]
            pos_dst = col[:n_e]

            # 负样本边（随机采样）
            num_negs = max(10, n_e)  # 推荐负样本数量和正样本差不多
            neg_src, neg_dst = [], []
            while len(neg_src) < num_negs:
                u = random.randint(0, graph_d.num_nodes() - 1)
                v = random.randint(0, graph_d.num_nodes() - 1)
                if not ((u, v) in zip(pos_src, pos_dst)):  # 避免采到正样本
                    neg_src.append(u)
                    neg_dst.append(v)

            # 合并正负边
            all_src = np.concatenate([pos_src, neg_src])
            all_dst = np.concatenate([pos_dst, neg_dst])
            y = np.concatenate([np.ones(n_e), np.zeros(num_negs)])

            # 构造 label 和 index
            graph_d.edge_label = torch.Tensor(y)
            graph_d.edge_label_index = torch.LongTensor([all_src.tolist(), all_dst.tolist()])

            graph_l.append(graph_d)




        graph_l = [graph.to(device) for graph in graph_l]
        random.shuffle(graph_l)

        model = WinGNN.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        model = model.to(device)

        n = math.ceil(len(graph_l) * 0.7)
        n_test = len(graph_l) - n

        best_param = train(args, model, optimizer, device, graph_l, global_state, logger, n)

        model.load_state_dict(best_param['best_state'])
        S_dw = best_param['best_s_dw']

        avg_mrr = test(graph_l, global_state, model, args, logger, n_test, S_dw, device)

        if avg_mrr > best_mrr:
            best_model = best_param['best_state']
        all_mrr_avg += avg_mrr

    torch.save(best_model, 'model_parameter/' + args.dataset + '.pkl')
    all_mrr_avg /= args.repeat
    print(all_mrr_avg)
