import os
from sklearn.metrics import pairwise


def get_node_sim(real_sim_matrix, adj):
    node_num = real_sim_matrix.shape[0]
    real_sim_list = torch.zeros(node_num, device=real_sim_matrix.device)
    for i in range(node_num):
        neighbor_indices = (adj[i] == 1).nonzero(as_tuple=False).squeeze()
        neighbor_num = neighbor_indices.numel()
        if neighbor_num == 0:
            real_sim_list[i] = torch.tensor(0, device=real_sim_matrix.device)
            continue
        cur_real = real_sim_matrix[i]
        cur_real_abs = torch.abs(cur_real)
        sorted_real_similarity, _ = torch.sort(cur_real_abs, descending=True)  # 对绝对值排序
        select_real = sorted_real_similarity[:neighbor_num].mean()
        real_sim_list[i] = select_real
    return real_sim_list


def compute_avg_cosine_similarity(adj, attr, label):
    dis_sim_list = []
    for i in range(adj.shape[0]):
        neighbor_indices = np.where(adj[i] == 1)[0]
        total_similarity = []
        num_neighbor = 0
        for j in neighbor_indices:
            similarity = pairwise.cosine_similarity([attr[i]], [attr[j]])
            similarity = similarity[0][0]
            num_neighbor += 1
            total_similarity.append(similarity)
        if num_neighbor > 0:
            total_similarity.sort()
            sim_len = math.floor(num_neighbor * 0.1)
            mean = np.mean(total_similarity[-sim_len:])
            dis_sim_list.append(mean)
        else:
            dis_sim_list.append(1)
    return dis_sim_list


def get_node_sim(sim_matrix, neighbor_mask):
    node_num = sim_matrix.shape[0]
    sim_list = torch.zeros(node_num, device=sim_matrix.device)
    for i in range(node_num):
        neighbors = neighbor_mask[i].nonzero(as_tuple=False).squeeze()
        if neighbors.numel() == 0:
            sim_list[i] = 0.0
            continue
        sim_values = sim_matrix[i, neighbors]
        sim_list[i] = sim_values.mean()
    return sim_list


def get_2hop_neighbor_mask(adj):
    device = adj.device
    n_nodes = adj.shape[0]
    hop1_mask = (adj == 1)
    self_mask = torch.eye(n_nodes, dtype=bool, device=device)
    adj_float = adj.float()
    adj_squared = torch.matmul(adj_float, adj_float)
    hop2_possible = (adj_squared > 0)
    hop2_mask = hop2_possible & (~hop1_mask) & (~self_mask)
    return hop2_mask


import numpy as np
import math

import torch
import utils


def get_2hop_neighbor_mask(adj):
    device = adj.device
    n_nodes = adj.shape[0]
    hop1_mask = (adj == 1)
    self_mask = torch.eye(n_nodes, dtype=bool, device=device)
    adj_float = adj.float()
    adj_squared = torch.matmul(adj_float, adj_float)
    hop2_possible = (adj_squared > 0)
    hop2_mask = hop2_possible & (~hop1_mask) & (~self_mask)
    return hop2_mask


def compute_neighbor_avg_sim(attrs, neighbor_mask, batch_size=1000):
    device = attrs.device
    n_nodes = attrs.shape[0]
    avg_sim = torch.zeros(n_nodes, device=device)

    for i in range(0, n_nodes, batch_size):
        batch_end = min(i + batch_size, n_nodes)
        batch_indices = torch.arange(i, batch_end, device=device)

        batch_mask = neighbor_mask[batch_indices]  # (batch_size, n_nodes)

        for idx in range(batch_indices.shape[0]):
            node_idx = batch_indices[idx]
            neighbors = batch_mask[idx].nonzero(as_tuple=False).squeeze()
            if neighbors.numel() == 0:
                avg_sim[node_idx] = 0.0
                continue
            if neighbors.dim() == 0:
                neighbors = neighbors.unsqueeze(0)

            node_feat = attrs[node_idx].unsqueeze(0)  # (1, feat_dim)
            neighbor_feats = attrs[neighbors]  # (num_neighbors, feat_dim)
            sims = torch.nn.functional.cosine_similarity(node_feat, neighbor_feats, dim=1)
            avg_sim[node_idx] = sims.mean()

    return avg_sim


def calculate_hop_similarities(adj_label, attrs):
    hop1_mask = (adj_label == 1).bool()
    hop2_mask = get_2hop_neighbor_mask(adj_label)

    first_hop_avg = compute_neighbor_avg_sim(attrs, hop1_mask, batch_size=2000)

    second_hop_avg = compute_neighbor_avg_sim(attrs, hop2_mask, batch_size=2000)

    return first_hop_avg, second_hop_avg


def get_2hop_neighbor_mask(adj):
    device = adj.device
    n_nodes = adj.shape[0]

    hop1_mask = (adj == 1).bool()

    self_mask = torch.eye(n_nodes, dtype=bool, device=device)

    adj_float = adj.float()
    adj_squared = torch.matmul(adj_float, adj_float)
    hop2_possible = (adj_squared > 0)

    hop2_mask = hop2_possible & (~hop1_mask) & (~self_mask)
    return hop1_mask, hop2_mask


def compute_hop_avg_sim(attrs, neighbor_mask, batch_size=500):
    device = attrs.device
    n_nodes = attrs.shape[0]
    avg_sim = torch.zeros(n_nodes, device=device)

    for batch_start in range(0, n_nodes, batch_size):
        batch_end = min(batch_start + batch_size, n_nodes)
        batch_idx = torch.arange(batch_start, batch_end, device=device)

        for i in batch_idx:

            neighbors = torch.nonzero(neighbor_mask[i], as_tuple=True)[0]
            if len(neighbors) == 0:
                avg_sim[i] = 0.0
                continue

            node_feat = attrs[i].unsqueeze(0)
            neighbor_feats = attrs[neighbors]
            sim = torch.nn.functional.cosine_similarity(node_feat, neighbor_feats, dim=1)
            avg_sim[i] = sim.mean()

    return avg_sim


def get_2hop_neighbor_mask(adj: torch.Tensor) -> torch.Tensor:
    device = adj.device
    n_nodes = adj.shape[0]

    hop1_mask = (adj == 1).bool()

    self_mask = torch.eye(n_nodes, dtype=bool, device=device)

    adj_float = adj.float()
    adj_squared = torch.matmul(adj_float, adj_float)
    hop2_possible = (adj_squared > 0).bool()

    hop2_mask = hop2_possible & (~hop1_mask) & (~self_mask)
    return hop2_mask


def get_2hop_similarity(adj: torch.Tensor, attrs: torch.Tensor, dataset_name) -> torch.Tensor:
    device = attrs.device
    node_num = attrs.shape[0]

    hop2_mask = get_2hop_neighbor_mask(adj)

    save_dir = './similarity_matrix'
    os.makedirs(save_dir, exist_ok=True)
    file_path = f'{save_dir}/2hop_similarity_matrix_{dataset_name}.pt'

    try:
        sim_matrix = torch.load(file_path)
        print(f"2-hop similarity matrix loaded from {file_path}")
        return sim_matrix
    except FileNotFoundError:
        print(f"Calculating 2-hop similarity matrix...")

    sim_matrix = torch.zeros((node_num, node_num), device=device)

    for i in range(node_num):

        hop2_indices = hop2_mask[i].nonzero(as_tuple=False).squeeze()
        if hop2_indices.numel() == 0:
            continue

        current_feat = attrs[i].unsqueeze(0).repeat(hop2_indices.numel(), 1)
        hop2_feats = attrs[hop2_indices]
        similarities = torch.cosine_similarity(current_feat, hop2_feats, dim=1)

        sim_matrix[i, hop2_indices] = similarities
        sim_matrix[hop2_indices, i] = similarities

    torch.save(sim_matrix, file_path)
    print(f"2-hop similarity matrix saved to {file_path}")
    return sim_matrix


def calculate_fused_diff_sim(adj, attrs, label, dataset_name, w1=1, w2=1, gamma=1.0):
    first_order_sim_matrix = utils.get_attr_similarity(adj, attrs, dataset_name)
    first_order_sim = get_node_sim(first_order_sim_matrix, adj)
    first_order_sim_np = first_order_sim.detach().cpu().numpy()

    hop2_mask = get_2hop_neighbor_mask(adj)
    second_order_sim_matrix = get_2hop_similarity(adj, attrs, dataset_name)
    second_order_sim = get_node_sim(second_order_sim_matrix, hop2_mask.float())
    second_order_sim_np = second_order_sim.detach().cpu().numpy()

    d_base = 1 - (w1 * first_order_sim_np + w2 * second_order_sim_np) / 2
    penalty = gamma * first_order_sim_np * (1 - second_order_sim_np)
    fused_diff = d_base + penalty

    return fused_diff


def get_node_sim(sim_matrix, neighbor_mask):
    node_num = sim_matrix.shape[0]
    sim_list = torch.zeros(node_num, device=sim_matrix.device)
    for i in range(node_num):
        neighbors = neighbor_mask[i].nonzero(as_tuple=False).squeeze()
        if neighbors.numel() == 0:
            sim_list[i] = 0.0
            continue
        sim_values = sim_matrix[i, neighbors]
        sim_list[i] = sim_values.mean()
    return sim_list
