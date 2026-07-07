import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tqdm
import torch
import dgl
import random
import math
import time
import numpy as np
from copy import deepcopy
from model.loss import prediction, Link_loss_meta
from model.utils import report_rank_based_eval_meta
import torch.nn.functional as F

def train(args, model, optimizer, device, graph_l, global_state, logger, n):
    model.to(device)
    best_param = {'best_mrr': 0, 'best_state': None}
    earl_stop_c = 0
    epoch_count = 0

    for epoch in range(args.epochs):
        random.seed(epoch)
        torch.manual_seed(epoch)
        np.random.seed(epoch)
        random.shuffle(graph_l)

        total_loss = torch.tensor(0.0, device=device)
        all_mrr = 0.0

        fast_weights = [p.clone().detach().to(device).requires_grad_(True) for p in model.parameters()]
        S_dw = [torch.zeros_like(p, device=device) for p in fast_weights]

        # 计算历史快照相似性权重 γ
        gamma_list = []
        for t in range(n):
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

        for t in range(n):
            graph = graph_l[t].to(device)
            features = graph.node_feature.to(device)


            graph.edge_label = graph.edge_label.to(device)
            # 🚀 确保 edge_label 既有 0 也有 1
            unique_labels = torch.unique(graph.edge_label)
            if len(unique_labels) == 1:
                print(f"⚠️ Warning: Only one class in edge_label at step {t}. Fixing it...")
                if unique_labels[0] == 1:
                    graph.edge_label[0] = 0  # 强制加入一个负样本
                else:
                    graph.edge_label[0] = 1  # 强制加入一个正样本
                print(f"✅ Fixed edge_label: {torch.unique(graph.edge_label)}")

            pred = model(graph, features, fast_weights)

            # 确保 `pred` 和 `graph.edge_label` 形状匹配
            min_len = min(pred.shape[0], graph.edge_label.shape[0])
            pred = pred[:min_len]
            graph.edge_label = graph.edge_label[:min_len]

            loss = Link_loss_meta(pred, graph.edge_label)

            # 计算 EMA 平滑梯度 ∇L_t^EMA
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
            beta = args.beta

            for i in range(len(S_dw)):
                if S_dw[i].shape != grad[i].shape:
                    S_dw[i] = torch.zeros_like(grad[i], device=device)

            S_dw = [beta * p[1] + (1 - beta) * p[0].pow(2) for p in zip(grad, S_dw)]
            adjusted_gradient = [p[0] / (torch.sqrt(p[1]) + 1e-8) for p in zip(grad, S_dw)]

            w = args.weight_param
            atg_gradient = [w * ag.clone() for ag in adjusted_gradient]

            if t > 0:
                for j in range(t):
                    gamma_value = float(gamma_list[t][j])  # 确保是标量
                    for i in range(len(atg_gradient)):  # 确保索引 i 适用于参数维度
                        if i >= len(adjusted_gradient):  # 防止索引超界
                            print(f"Error: adjusted_gradient index {i} out of range")
                            continue

                        if adjusted_gradient[i].shape != atg_gradient[i].shape:
                            print(f"Shape mismatch: adjusted_gradient[{i}].shape = {adjusted_gradient[i].shape}, "
                                  f"atg_gradient[{i}].shape = {atg_gradient[i].shape}")

                            # 计算填充量，确保最小修改
                            pad_right = max(0, atg_gradient[i].shape[1] - adjusted_gradient[i].shape[1]) if \
                            adjusted_gradient[i].dim() > 1 else 0
                            pad_bottom = max(0, atg_gradient[i].shape[0] - adjusted_gradient[i].shape[0])

                            # 仅填充不同部分
                            adjusted_gradient[i] = F.pad(adjusted_gradient[i], (0, pad_right, 0, pad_bottom))

                        atg_gradient[i] += (1 - w) * gamma_value * adjusted_gradient[i]

                for i in range(len(atg_gradient)):
                    if adjusted_gradient[i].shape != atg_gradient[i].shape:
                        if adjusted_gradient[i].numel() == atg_gradient[i].numel():
                            adjusted_gradient[i] = adjusted_gradient[i].view(atg_gradient[i].shape)
                        else:
                            pad_right = max(0, atg_gradient[i].shape[1] - adjusted_gradient[i].shape[1]) if \
                            adjusted_gradient[i].dim() > 1 else 0
                            pad_bottom = max(0, atg_gradient[i].shape[0] - adjusted_gradient[i].shape[0])
                            adjusted_gradient[i] = F.pad(adjusted_gradient[i], (0, pad_right, 0, pad_bottom))

            mrr, *_ = report_rank_based_eval_meta(model, graph, features, fast_weights)
            all_mrr += mrr
            total_loss += loss

        avg_mrr = all_mrr / n
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        mrr, *_ = report_rank_based_eval_meta(model, graph, features, fast_weights)
        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph.edge_label)

        # 如果预测不可用（label 单一），不要记入日志
        if acc == -1 and ap == -1:
            logger.info(f"meta epoch:{epoch}, mrr:{avg_mrr:.5f}, loss:{total_loss.item():.5f},  ！指标跳过（label 单一）")
        else:
            logger.info(
                f"meta epoch:{epoch}, mrr:{avg_mrr:.5f}, loss:{total_loss.item():.5f}, acc:{acc:.5f}, ap:{ap:.5f}, f1:{f1:.5f}, macro_auc:{macro_auc:.5f}, micro_auc:{micro_auc:.5f}")

        epoch_count += 1
        if avg_mrr > best_param['best_mrr']:
            best_param = {
                'best_mrr': avg_mrr,
                'best_state': deepcopy(model.state_dict()),
                'best_s_dw': deepcopy(S_dw)  # ✅ 确保 best_s_dw 被保存
            }

            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break

        with torch.no_grad():
            for param, fast_weight in zip(model.parameters(), fast_weights):
                param.copy_(fast_weight)

    return best_param
