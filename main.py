from local_analysis import calculate_fused_diff_sim
import scipy as sp
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

from model import Model, Attribute_Decoder, Structure_Decoder
from utils import load_anomaly_detection_dataset, polynomial_pacing_function, edge_index_to_adjacency, normalize_adj, \
    precision_top_k, recall_top_k, f1_score_top_k, load_npz, random_seed
import scipy.sparse as sp
import scipy.io as sio


def loss_func(adj, A_hat, attrs, X_hat, real_sim, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sum(diff_attribute, 1)
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sum(diff_structure, 1)
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def save_data(adj, attrs, label, filename):
    adj_csc = sp.csc_matrix(adj)
    attrs_csc = sp.csc_matrix(attrs)
    label_np = np.array(label)
    sio.savemat(filename + '.mat', {'Network': adj_csc, 'Attributes': attrs_csc, 'Label': label_np})


def train(args):
    if args.dataset in ['weibo', 'reddit', 'reddit', 'disney', 'books', 'inj_cora']:
        from pygod.utils import load_data
        data = load_data(args.dataset)
        attrs = data.x.numpy().astype(np.float32)
        init_adj = data.edge_index
        adj_origin = edge_index_to_adjacency(init_adj, len(attrs))
        np.fill_diagonal(adj_origin, 1)
        adj_label = adj_origin
        adj = normalize_adj(adj_origin).toarray()
        label = data.y.bool()
        label = torch.where(label, 1.0, 0.0).numpy()
        save_data(adj, attrs, label, args.dataset)
    elif args.dataset in ['tolokers', 'questions']:
        adj, attrs, label, adj_label = load_npz(args.dataset)
    else:
        adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)

    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    adj_noself = adj_label.clone()
    adj_noself.fill_diagonal_(0)
    real_sim = calculate_fused_diff_sim(adj_noself, attrs, label, args.dataset)
    sorted_sim_index = np.argsort(real_sim)
    num_elements = int(args.ratio * len(real_sim))
    pre_train_index = sorted_sim_index[:num_elements]

    model_pre = Model(feat_size=attrs.size(1), hidden_size=args.hidden_dim, dropout=args.dropout)
    best_auc = 0.0
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = attrs.to(device)
        model_pre = model_pre.cuda()

    optimizer_pre = torch.optim.Adam(model_pre.parameters(), lr=args.lr)

    T = args.epoch
    K_norm = len(pre_train_index)
    for epoch in range(args.epoch):
        norm_len = polynomial_pacing_function(epoch + 1, T, K_norm, lambda_param=args.lambda_param)
        norm_index = pre_train_index[:norm_len]
        sub_adj = adj[norm_index, :][:, norm_index]
        sub_adj_label = adj_label[norm_index, :][:, norm_index]
        sub_attrs = attrs[norm_index, :]

        model_pre.train()
        optimizer_pre.zero_grad()
        A_hat, X_hat = model_pre(sub_attrs, sub_adj)
        loss, struct_loss, feat_loss = loss_func(sub_adj_label, A_hat, sub_attrs, X_hat, real_sim[norm_index],
                                                 args.alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer_pre.step()
        print("Pre-Training Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=",
              "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))

    model_pre.eval()
    A_hat, X_hat = model_pre(attrs, adj)
    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, real_sim, args.alpha)
    score = loss.detach().cpu().numpy()
    auc = roc_auc_score(label, score)
    print("Pre-Training Epoch:", '%04d' % (epoch), 'Dataset:', args.dataset, 'Auc', auc)


    for param in model_pre.shared_encoder.parameters():
        param.requires_grad = False

    model_pre.attr_decoder = Attribute_Decoder(attrs.size(1), args.hidden_dim, dropout=args.dropout).to(
        torch.device(args.device))
    model_pre.struct_decoder = Structure_Decoder(args.hidden_dim, dropout=args.dropout).to(torch.device(args.device))


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_pre.parameters()), lr=args.lr)

    sorted_sim_index = np.argsort(-real_sim)
    num_elements = int(args.ratio * len(real_sim))
    train_index = sorted_sim_index[:num_elements]
    K_anom = len(train_index)
    for epoch in range(args.epoch):
        anom_len = polynomial_pacing_function(epoch + 1, T, K_anom, lambda_param=args.lambda_param)
        anom_index = train_index[:anom_len]

        sub_adj = adj[anom_index, :][:, anom_index]
        sub_adj_label = adj_label[anom_index, :][:, anom_index]
        sub_attrs = attrs[anom_index, :]

        model_pre.train()
        optimizer.zero_grad()
        A_hat, X_hat = model_pre(sub_attrs, sub_adj)
        loss, struct_loss, feat_loss = loss_func(sub_adj_label, A_hat, sub_attrs, X_hat, real_sim[anom_index],
                                                 args.alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=",
              "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        if (epoch + 1) % 5 == 0:
            model_pre.eval()
            A_hat, X_hat = model_pre(attrs, adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, real_sim, args.alpha)
            score = loss.detach().cpu().numpy()
            auc = roc_auc_score(label, score)
            precision = precision_top_k(score, label, [100])
            recall = recall_top_k(score, label, [100])
            f1_score = f1_score_top_k(score, label, [100])


            if best_auc < auc:
                best_auc = auc
                print('---------Best-Auc:', best_auc)
                print("Epoch:", '%04d' % (epoch), 'Dataset:', args.dataset, 'Auc', best_auc, "Recall:", recall,
                      "Precision:",
                      precision, "F1-score:", f1_score)
    print("----------------------------", 'Dataset:', args.dataset, 'Auc', best_auc,"----------------------------"  )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='YelpRes')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--seed', type=int, default=10, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--ratio', type=float, default=0.25, help='balance parameter')
    parser.add_argument('--lambda_param', type=float, default=1.0, help='pacing function exponent')
    args = parser.parse_args()

    print("Random-seed: {}".format(args.seed))
    random_seed(args.seed)
    train(args)
