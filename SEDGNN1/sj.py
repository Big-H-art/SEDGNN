import numpy as np

# 替换为您本地的文件路径
file_path = 'D:/cfx/SEDGNN1/WinGNN-main/dataset/uci-msg/edge_feature/1.npy'


# 加载文件内容
data = np.load(file_path)

# 打印数据的维度和前几行内容
print("数据形状:", data.shape)
print("前5行数据:")
print(data[:100000])
