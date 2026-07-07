#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : Load and preprocess datasets
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

def preprocess_dataset(csv_file, output_dir, num_snapshots):
    """
    预处理数据集，将其划分为多个时间快照并保存。
    :param csv_file: 输入的 CSV 数据集路径
    :param output_dir: 预处理后输出的文件夹路径
    :param num_snapshots: 快照数量
    """
    # 加载 CSV 数据
    data = pd.read_csv(csv_file, header=None, names=['source', 'target', 'rating', 'timestamp'])

    # 按时间戳排序
    data = data.sort_values(by='timestamp').reset_index(drop=True)

    # 计算时间范围和每个快照的间隔
    min_time = data['timestamp'].min()
    max_time = data['timestamp'].max()
    time_interval = (max_time - min_time) / num_snapshots

    # 创建输出目录和子文件夹
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edge_index"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "node_feature"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edge_feature"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edge_time"), exist_ok=True)

    # 节点映射
    nodes = sorted(set(data['source']).union(set(data['target'])))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    nodes_num = len(nodes)

    # 按时间划分快照
    for i in range(num_snapshots):
        # 计算快照的时间范围
        start_time = min_time + i * time_interval
        end_time = start_time + time_interval

        # 获取当前快照的数据
        snapshot = data[(data['timestamp'] >= start_time) & (data['timestamp'] < end_time)]

        # 如果当前快照没有数据，则跳过
        if snapshot.empty:
            continue

        # 边索引 (source 和 target 转换为索引)
        edge_index = np.array([
            snapshot['source'].map(node_to_idx).values,
            snapshot['target'].map(node_to_idx).values
        ])
        np.save(os.path.join(output_dir, "edge_index", f"{i}.npy"), edge_index)

        # 节点特征 (全 1，占位特征)
        node_feature = np.ones((nodes_num, 1))
        np.save(os.path.join(output_dir, "node_feature", f"{i}.npy"), node_feature)

        # 边特征 (rating 列作为特征)
        edge_feature = snapshot['rating'].values.reshape(-1, 1)
        np.save(os.path.join(output_dir, "edge_feature", f"{i}.npy"), edge_feature)

        # 边时间 (timestamp 列作为时间特征)
        edge_time = snapshot['timestamp'].values
        np.save(os.path.join(output_dir, "edge_time", f"{i}.npy"), edge_time)

    print(f"数据集预处理完成！文件已保存至 {output_dir}")

def load_r(processed_dir):
    """
    加载预处理后的快照数据
    :param processed_dir: 已处理数据的文件夹路径
    :return: 子图列表、边特征、边时间、节点特征
    """
    # 定义各子文件夹路径
    path_ei = os.path.join(processed_dir, 'edge_index')
    path_nf = os.path.join(processed_dir, 'node_feature')
    path_ef = os.path.join(processed_dir, 'edge_feature')
    path_et = os.path.join(processed_dir, 'edge_time')

    # 获取文件列表并排序
    edge_index_files = sorted(os.listdir(path_ei))
    node_feature_files = sorted(os.listdir(path_nf))
    edge_feature_files = sorted(os.listdir(path_ef))
    edge_time_files = sorted(os.listdir(path_et))

    # 初始化返回值
    sub_graphs = []
    edge_features = []
    edge_times = []
    node_features = []

    # 遍历每个快照的文件
    for i in range(len(edge_index_files)):
        # 加载边索引
        edge_index = np.load(os.path.join(path_ei, edge_index_files[i]))
        row, col = edge_index
        # 构造稀疏矩阵表示子图
        nodes_num = np.load(os.path.join(path_nf, node_feature_files[i])).shape[0]
        ts = [1] * len(row)  # 边权重，默认为 1
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graphs.append(sub_g)

        # 加载节点特征
        node_feature = np.load(os.path.join(path_nf, node_feature_files[i]))
        node_features.append(node_feature)

        # 加载边特征
        edge_feature = np.load(os.path.join(path_ef, edge_feature_files[i]))
        edge_features.append(edge_feature)

        # 加载边时间
        edge_time = np.load(os.path.join(path_et, edge_time_files[i]))
        edge_times.append(edge_time)

    return sub_graphs, edge_features, edge_times, node_features

if __name__ == "__main__":
    # 数据集路径
    csv_file = "D:/cfx/AGS-main/WinGNN-main/dataset/bitcoinotc.csv"

    # 输出文件夹路径
    output_dir = "D:/cfx/AGS-main/WinGNN-main/dataset/bitcoinotc"

    # 快照数量
    num_snapshots = 262

    # 执行数据预处理
    preprocess_dataset(csv_file, output_dir, num_snapshots)
