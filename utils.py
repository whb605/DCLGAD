import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn.metrics import precision_recall_curve, auc, pairwise


def get_adj(sparse_adjacency_matrix):
    max_index = max(max(index_pair) for index_pair in sparse_adjacency_matrix) + 1
    dimension = max_index  # 假设索引是从0开始的

    # 创建一个全零的二维矩阵
    adjacency_matrix = np.zeros((dimension, dimension), dtype=int)

    # 遍历稀疏矩阵中的每个元素
    for index_pair in sparse_adjacency_matrix:
        if index_pair.size == 2:  # 确保每个元素确实包含两个索引
            i, j = index_pair
            adjacency_matrix[i][j] = 1
    return adjacency_matrix


def load_npz(dataset):
    data = np.load("./dataset/{}.npz".format(dataset))
    feat = data['node_features']
    adj = get_adj(data['edges'])
    labels = data['node_labels']
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    return adj_norm, feat, labels, adj


def load_anomaly_detection_dataset(dataset, datadir='./dataset'):
    data = sio.loadmat(f'{datadir}/{dataset}.mat')
    truth = data['Label'] if ('Label' in data) else data['gnd']
    feat = data['Attributes'] if ('Attributes' in data) else data['X']
    adj = data['Network'] if ('Network' in data) else data['A']

    truth = truth.flatten()

    adj = sp.csr_matrix(adj)
    adj_self_loop = adj.copy()
    adj_self_loop.setdiag(1)
    adj_norm = normalize_adj(adj_self_loop).toarray()
    adj = adj.toarray()

    if dataset != 'Enron':
        feat = feat.toarray()
    return adj_norm, feat, truth, adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def exponential_pacing_function(t, T, K,lambda_param):

    value = (1 - math.exp(-lambda_param * (t / T))) * K
    return int(value)

def logarithmic_pacing_function(t, T, K):
    value = [1 + 0.1 * math.log((t / (T + math.exp(-10))) if T != 0 else 1)] * K
    return int(value)


def polynomial_pacing_function(t, T, K, lambda_param=1):
    value = ((t / T) ** lambda_param) * K
    return max(1, int(value))


def edge_index_to_adjacency(edge_index, num_nodes):
    # 创建一个全零的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 根据边索引列表更新邻接矩阵
    for edge in edge_index.T:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 如果是有向图，可以去掉这行

    return adjacency_matrix


def recall_top_k(y_pred, y_true, k_list):
    """
    Computes the recall for the top k elements
    """
    recall_list = []
    for k in k_list:
        top_k_indices = np.argsort(y_pred)[-k:]
        y_pred_top_k = np.zeros_like(y_pred)
        y_pred_top_k[top_k_indices] = 1
        recall = np.sum(np.logical_and(y_pred_top_k, y_true)) / np.sum(y_true)
        recall_list.append(recall)
    return recall_list


def precision_top_k(y_pred, y_true, k_list):
    """
    Computes the precision for the top k elements
    """
    precision_list = []
    for k in k_list:
        top_k_indices = np.argsort(y_pred)[-k:]
        y_pred_top_k = np.zeros_like(y_pred)
        y_pred_top_k[top_k_indices] = 1
        precision = np.sum(np.logical_and(y_pred_top_k, y_true)) / k
        precision_list.append(precision)
    return precision_list


def f1_score_top_k(y_pred, y_true, k_list):
    """
    Computes the F1 score for the top k elements
    """
    f1_score_list = []
    for k in k_list:
        top_k_indices = np.argsort(y_pred)[-k:]
        y_pred_top_k = np.zeros_like(y_pred)
        y_pred_top_k[top_k_indices] = 1

        true_positive = np.sum(np.logical_and(y_pred_top_k, y_true))
        precision = true_positive / k
        recall = true_positive / np.sum(y_true)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        f1_score_list.append(f1_score)
    return f1_score_list


def calculate_similarity(att_s, att_d, att_w=0):
    # # att_o 是两者都有的属性
    # att_o = np.logical_and(att_s != 0, att_d != 0).astype(float)
    # # 计算加权公有属性在原节点的占比
    # cnt_s = np.sum(att_s * att_w)
    # cnt_o = np.sum(att_o * att_w)
    # if cnt_s == 0: cnt_s = 1
    # return cnt_o / cnt_s
    similarity = pairwise.cosine_similarity([att_s], [att_d])
    similarity = similarity[0][0]
    return math.fabs(similarity)


# def get_attr_similarity(adj: np.ndarray, attrs: np.ndarray) -> np.ndarray:
#     similarity_matrix = np.zeros_like(adj)
#     node_num = adj.shape[0]
#     for i in range(node_num):
#         for j in range(i, node_num):
#             if adj[i, j] == 1:
#                 similarity = calculate_similarity(attrs[i], attrs[j])
#                 similarity_matrix[i, j] = similarity
#                 similarity_matrix[j, i] = similarity
#     return similarity_matrix

def save_similarity_matrix(similarity_matrix, file_path):
    # 保存相似度矩阵到本地文件
    torch.save(similarity_matrix, file_path)
    print(f"Similarity matrix saved to {file_path}")


def load_similarity_matrix(file_path):
    # 尝试从本地文件加载相似度矩阵
    try:
        similarity_matrix = torch.load(file_path)
        print(f"Similarity matrix loaded from {file_path}")
        return similarity_matrix
    except FileNotFoundError:
        print(f"No similarity matrix found at {file_path}. Please generate first.")
        return None


def cal_similarity(adj: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
    node_num = attrs.shape[0]
    similarity_matrix = torch.zeros((node_num, node_num), device=attrs.device)

    for i in range(node_num):
        neighbors_idx = torch.where(adj[i] == 1)[0]

        if neighbors_idx.shape[0] == 0:
            continue

        current_node_attr = attrs[i].unsqueeze(0).repeat(neighbors_idx.shape[0], 1)
        neighbors_attrs = attrs[neighbors_idx]

        similarities = torch.cosine_similarity(current_node_attr, neighbors_attrs, dim=1)

        similarity_matrix[i, neighbors_idx] = similarities
        similarity_matrix[neighbors_idx, i] = similarities

    return similarity_matrix


def get_attr_similarity(adj: torch.Tensor, attrs: torch.Tensor, dataset_name) -> torch.Tensor:
    node_num = attrs.shape[0]
    attr_dim = attrs.shape[1]
    similarity_matrix = torch.zeros((node_num, node_num), device=attrs.device)

    # 创建保存相似度矩阵的目录
    save_dir = './similarity_matrix'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 构造文件路径
    file_path = f'{save_dir}/similarity_matrix_{dataset_name}.pt'

    # 尝试从本地文件加载相似度矩阵
    try:
        similarity_matrix = torch.load(file_path)
        print(f"Similarity matrix loaded from {file_path}")
        return similarity_matrix
    except FileNotFoundError:
        print(f"No similarity matrix found at {file_path}. Calculating similarity matrix...")

    # 批量计算相似度，减少内存消耗
    for i in range(node_num):
        # 获取当前节点i的所有邻居节点的索引
        neighbors_idx = torch.where(adj[i] == 1)[0]

        # 如果没有邻居，跳过当前节点
        if neighbors_idx.shape[0] == 0:
            continue

        # 获取当前节点i的特征和其邻居的特征
        current_node_attr = attrs[i].unsqueeze(0).repeat(neighbors_idx.shape[0], 1)
        neighbors_attrs = attrs[neighbors_idx]

        # 计算当前节点与所有邻居的相似度
        similarities = torch.cosine_similarity(current_node_attr, neighbors_attrs, dim=1)

        # 将相似度存入相似度矩阵
        similarity_matrix[i, neighbors_idx] = similarities
        similarity_matrix[neighbors_idx, i] = similarities  # 填充对称部分

    torch.save(similarity_matrix, file_path)
    print(f"Similarity matrix saved to {file_path}")

    return similarity_matrix

def compute_global_similarity_mat(feat: torch.Tensor, dataset_name: str, batch_size=1000) -> torch.Tensor:

    # 1. 缓存路径设置
    save_dir = './similarity_matrix'
    os.makedirs(save_dir, exist_ok=True)
    cache_path = f'{save_dir}/global_similarity_matrix_{dataset_name}.pt'

    # 2. 尝试加载缓存（避免重复计算）
    try:
        global_sim_mat = torch.load(cache_path)
        print(f"Loaded global similarity matrix from cache: {cache_path}")
        return global_sim_mat
    except FileNotFoundError:
        print(f"No global similarity matrix cache. Calculating (batch_size={batch_size})...")

    # 3. 分批计算全量相似度（避免内存溢出）
    n_nodes = feat.shape[0]
    device = feat.device
    global_sim_mat = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)

    for i in range(0, n_nodes, batch_size):
        # 当前批次的特征（i ~ min(i+batch_size, n_nodes)）
        batch_end_i = min(i + batch_size, n_nodes)
        feat_batch_i = feat[i:batch_end_i]  # (batch_i, d)
        print(i)
        for j in range(i, n_nodes, batch_size):
            print(j)
            # 目标批次的特征（j ~ min(j+batch_size, n_nodes)）
            batch_end_j = min(j + batch_size, n_nodes)
            feat_batch_j = feat[j:batch_end_j]  # (batch_j, d)

            # 计算两批次之间的两两相似度（利用torch向量化加速）
            # cosine_similarity(feat_batch_i.unsqueeze(1), feat_batch_j.unsqueeze(0), dim=2) → (batch_i, batch_j)
            sim_batch = torch.nn.functional.cosine_similarity(
                feat_batch_i.unsqueeze(1),
                feat_batch_j.unsqueeze(0),
                dim=2
            )

            # 填充相似度矩阵（对称矩阵，i→j和j→i相同）
            global_sim_mat[i:batch_end_i, j:batch_end_j] = sim_batch
            global_sim_mat[j:batch_end_j, i:batch_end_i] = sim_batch.T  # 转置填充对称部分

    # 4. 保存缓存
    torch.save(global_sim_mat, cache_path)
    print(f"Saved global similarity matrix to cache: {cache_path}")
    return global_sim_mat

def cosine_similarity(feat1, feat2):
    # 确保feat1和feat2都是二维张量，且在特征维度上具有相同的大小
    return torch.nn.functional.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)

def compute_similarity_mat(feat, batch_size=1000):
    n = feat.shape[0]
    similarity_mat = torch.zeros((n, n), dtype=torch.float32)

    # 分批计算
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        feat_batch = feat[i:batch_end]

        # 计算当前批次与所有后续特征的相似度
        for j in range(i + batch_size, n, batch_size):
            next_batch_end = min(j + batch_size, n)
            feat_next_batch = feat[j:next_batch_end]
            similarity_value = cosine_similarity(feat_batch, feat_next_batch)  # 计算两个批次之间的相似度
            similarity_mat[i:batch_end, j:next_batch_end] = similarity_value
            similarity_mat[j:next_batch_end, i:batch_end] = similarity_value.T  # 转置以填充对称矩阵

    # 将对角线元素设置为0
    # torch.fill_diagonal_(similarity_mat, 0)

    return similarity_mat


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# def cosine_similarity(features, u, v):
#     dot_product = np.dot(features[u], features[v])
#     norm_u = np.linalg.norm(features[u])
#     norm_v = np.linalg.norm(features[v])
#     sim = dot_product / (norm_u * norm_v) if norm_u > 0 and norm_v > 0 else 0
#     return sim

def mask_anom(real_sim_matrix):
    sub_matrix = real_sim_matrix

    # 第一步：计算每一行不为0的值的平均值
    row_means = []
    for row in sub_matrix:
        non_zero_values = row[row != 0]
        if non_zero_values.numel() > 0:
            mean_value = non_zero_values.mean()
            row_means.append(mean_value)
        else:
            row_means.append(0)  # 如果该行全为0，平均值设为0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将row_means转换为与sub_matrix行数匹配的张量，方便后续比较操作
    row_means_tensor = torch.tensor(row_means).unsqueeze(1).expand(-1, sub_matrix.size(1)).to(device)

    # 第二步：将小于平均值的值替换为0
    sub_matrix = torch.where(sub_matrix < row_means_tensor, torch.tensor(0).to(device), sub_matrix)
    return sub_matrix



def get_attr_similarity(adj: torch.Tensor, attrs: torch.Tensor, dataset_name) -> torch.Tensor:
    node_num = attrs.shape[0]
    attr_dim = attrs.shape[1]
    similarity_matrix = torch.zeros((node_num, node_num), device=attrs.device)

    # 创建保存相似度矩阵的目录
    save_dir = './similarity_matrix'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 构造文件路径
    file_path = f'{save_dir}/similarity_matrix_{dataset_name}.pt'

    # 尝试从本地文件加载相似度矩阵
    try:
        similarity_matrix = torch.load(file_path)
        print(f"Similarity matrix loaded from {file_path}")
        return similarity_matrix
    except FileNotFoundError:
        print(f"No similarity matrix found at {file_path}. Calculating similarity matrix...")

    # 批量计算相似度，减少内存消耗
    for i in range(node_num):
        # 获取当前节点i的所有邻居节点的索引
        neighbors_idx = torch.where(adj[i] == 1)[0]

        # 如果没有邻居，跳过当前节点
        if neighbors_idx.shape[0] == 0:
            continue

        # 获取当前节点i的特征和其邻居的特征
        current_node_attr = attrs[i].unsqueeze(0).repeat(neighbors_idx.shape[0], 1)
        neighbors_attrs = attrs[neighbors_idx]

        # 计算当前节点与所有邻居的相似度
        similarities = torch.cosine_similarity(current_node_attr, neighbors_attrs, dim=1)

        # 将相似度存入相似度矩阵
        similarity_matrix[i, neighbors_idx] = similarities
        similarity_matrix[neighbors_idx, i] = similarities  # 填充对称部分

    torch.save(similarity_matrix, file_path)
    print(f"Similarity matrix saved to {file_path}")

    return similarity_matrix