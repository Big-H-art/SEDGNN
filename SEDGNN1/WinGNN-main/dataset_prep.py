import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch


def load(nodes_num):
    """
    load_dataset
    :param nodes_num:
    :return:
    """
    path = "dataset/uci-msg/"

    train_e_feat_path = path + 'train_e_feat/'
    test_e_feat_path = path + 'test_e_feat/'

    train_n_feat_path = path + 'train_n_feat/'
    test_n_feat_path = path + 'test_n_feat/'

    train_path = path + '/train/'
    test_path = path + '/test/'

    train_n_feat = read_e_feat(train_n_feat_path)
    test_n_feat = read_e_feat(test_n_feat_path)

    train_e_feat = read_e_feat(train_e_feat_path)
    test_e_feat = read_e_feat(test_e_feat_path)

    num = 0
    train_graph = read_graph(train_path, nodes_num, num)
    num = num + len(train_graph)
    test_graph = read_graph(test_path, nodes_num, num)

    return train_graph, train_e_feat, train_n_feat, test_graph, test_e_feat, test_n_feat


def load_r(name):
    path = os.path.join("dataset", name)
    path_gs = os.path.join(path, "global_state")
    os.makedirs(path_gs, exist_ok=True)

    edge_index = read_npz(os.path.join(path, 'edge_index'))
    edge_feature = read_npz(os.path.join(path, 'edge_feature'))
    node_feature = read_npz(os.path.join(path, 'node_feature'))
    edge_time = read_npz(os.path.join(path, 'edge_time'))

    global_state_path = os.path.join(path_gs, "global_state.npy")
    if os.path.exists(global_state_path):
        global_state = np.load(global_state_path)
        print(f"✅ global_state 加载成功，形状 = {global_state.shape}")
    else:
        print(f"❌ global_state.npy 文件不存在，计算并保存 global_state")
        global_state = generate_global_state(node_feature)  # 传入节点特征
        save_global_state(path, global_state)

    nodes_num = node_feature[0].shape[0]
    sub_graph = [coo_matrix((np.ones(len(e_i[0])), (e_i[0], e_i[1])), shape=(nodes_num, nodes_num)) for e_i in
                 edge_index]

    return sub_graph, edge_feature, edge_time, node_feature, global_state


def save_global_state(output_dir, global_state):
    global_state_dir = os.path.join(output_dir, "global_state")
    try:
        os.makedirs(global_state_dir, exist_ok=True)
        print(f"📁 Debug: global_state 目录创建成功 -> {global_state_dir}")
    except Exception as e:
        print(f"❌ 目录创建失败: {e}")
        return

    if len(global_state) == 0:
        print(f"⚠️ Warning: global_state 为空，未保存文件！")
        return

    global_state_path = os.path.join(global_state_dir, "global_state.npy")
    np.save(global_state_path, global_state)
    print(f"✅ global_state 已保存，路径: {global_state_path}")


def generate_global_state(node_feature):
    """
    计算全局图状态 (global_state)：
    g_t = mean(H_t)

    :param node_feature: List[np.ndarray]，每个快照的节点特征矩阵 (n, d)
    :return: np.ndarray，形状 (T, d)，每个时间快照的 global_state
    """
    global_state = []
    for H_t in node_feature:
        if H_t.shape[0] == 0:  # 如果当前快照没有节点特征
            global_state.append(np.zeros(H_t.shape[1]))  # 设为 0 向量
        else:
            avg_Ht = np.mean(H_t, axis=0)  # 计算均值池化
            global_state.append(avg_Ht)
    return np.array(global_state)  # 返回所有快照的 global_state


def read_npz(path):
    if not os.path.exists(path):
        print(f"❌ Warning: {path} does not exist. Returning empty list.")
        return []
    files = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))
    if len(files) == 0:
        print(f"⚠️ Warning: {path} 目录为空，没有任何 .npz 文件！")
    return [np.load(os.path.join(path, filename)) for filename in files]


def read_e_feat(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Returning empty list.")
        return []
    return [np.load(os.path.join(path, filename)) for filename in
            sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))]


def read_graph(path, nodes_num, num):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Returning empty graph list.")
        return []
    sub_graph = []
    for file in sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]) - num):
        sub_ = pd.read_csv(os.path.join(path, file))
        row, col = sub_['src_l'].values, sub_['dst_l'].values
        sub_graph.append(coo_matrix((np.ones(len(row)), (row, col)), shape=(nodes_num, nodes_num)))
    return sub_graph