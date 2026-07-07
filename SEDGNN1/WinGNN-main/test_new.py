import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
from copy import deepcopy
from model.loss import prediction, Link_loss_meta
from model.utils import report_rank_based_eval_meta
import torch.nn.functional as F

def test(graph_l, global_state, model, args, logger, n, S_dw, device):
    model.eval()
    model.to(device)

    beta = args.beta  # EMA平滑系数
    avg_mrr = 0.0
    avg_auc = 0.0
    avg_acc = 0.0

    # 初始化模型参数快照
    fast_weights = [p.clone().detach().to(device).requires_grad_(True) for p in model.parameters()]
    S_dw = [torch.zeros_like(p, device=device) for p in fast_weights]
    historical_gradients = []  # 存储历史快照梯度

    # 计算历史快照相似性权重
    gamma_list = []
    for t in range(n, len(graph_l)):
        g_t = torch.tensor(global_state[t], dtype=torch.float32, device=device)
        similarities = []
        for j in range(t):
            g_j = torch.tensor(global_state[j], dtype=torch.float32, device=device)
            cos_sim = F.cosine_similarity(g_t, g_j, dim=0)
            D_tj = 1 - cos_sim
            similarities.append(D_tj.item())

        if similarities:
            similarities = torch.tensor(similarities, dtype=torch.float32, device=device)
            gamma_t = torch.softmax(-similarities, dim=0).tolist()
        else:
            gamma_t = []
        gamma_list.append(gamma_t)

    for t in range(n, len(graph_l)):
        graph = graph_l[t].to(device)
        features = graph.node_feature.to(device)
        graph.edge_label = graph.edge_label.to(device)

        # 前向传播
        pred = model(graph, features, fast_weights)
        pred = pred.view(-1)
        if pred.shape[0] != graph.edge_label.shape[0]:
            min_len = min(pred.shape[0], graph.edge_label.shape[0])
            pred = pred[:min_len]
            graph.edge_label = graph.edge_label[:min_len]

        loss = Link_loss_meta(pred, graph.edge_label)

        with torch.enable_grad():
            grad = torch.autograd.grad(loss, fast_weights, create_graph=False)
            S_dw = [beta * p[1] + (1 - beta) * p[0].pow(2) for p in zip(grad, S_dw)]
            adjusted_gradient = [p[0] / (torch.sqrt(p[1]) + 1e-8) for p in zip(grad, S_dw)]

            w = args.weight_param
            atg_gradient = [w * ag for ag in adjusted_gradient]  # 计算 ∇L_t^EMA

            if t > n:
                for j in range(n, t):
                    adjusted_gradient_j = historical_gradients[j - n]  # 取历史快照 j 的 ∇L_j^EMA
                    for i in range(len(atg_gradient)):
                        atg_gradient[i] += (1 - w) * gamma_list[t - n][j - n] * adjusted_gradient_j[i]
                fast_weights = [p - args.maml_lr * cg for p, cg in zip(fast_weights, atg_gradient)]
            else:
                fast_weights = [p - args.maml_lr * cg for p, cg in zip(fast_weights, atg_gradient)]

        historical_gradients.append(adjusted_gradient)  # 存储当前时间步的梯度
        mrr, *_ = report_rank_based_eval_meta(model, graph, features, fast_weights)
        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph.edge_label)

        avg_mrr += mrr
        avg_auc += macro_auc
        avg_acc += acc

        logger.info(f'Test MRR: {mrr:.5f}, AUC: {macro_auc:.5f}, Acc: {acc:.5f}, AP: {ap:.5f}, '
                    f'F1: {f1:.5f}, Macro AUC: {macro_auc:.5f}, Micro AUC: {micro_auc:.5f}')

    num_test_steps = len(graph_l) - n
    avg_mrr /= num_test_steps
    avg_auc /= num_test_steps
    avg_acc /= num_test_steps

    logger.info({'avg_acc': avg_acc, 'avg_auc': avg_auc, 'avg_mrr': avg_mrr})

    return avg_mrr
