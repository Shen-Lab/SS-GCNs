import torch
import torch.nn as nn
import numpy as np
import random

import argparse

import net as net
from utils import load_data, load_data_raw, graph_attack, preprocess_feat_adj
from sklearn.metrics import f1_score


def run(args, seed):

    setup_seed(seed)
    dataset = args['dataset']
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset) 
    adj = adj.cuda()

    features = features.cuda()
    labels = labels.cuda()

    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'])
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loss_func = nn.CrossEntropyLoss()
    loss_val = []
    early_stopping = 10

    for epoch in range(1000):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss_train = loss_func(output[idx_train], labels[idx_train])
        # print('epoch', epoch, 'loss', loss_train.data)
        loss_train.backward()
        optimizer.step()

        # validation
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            loss_val.append(loss_func(output[idx_val], labels[idx_val]).cpu().numpy())
            # print('val acc', f1_score(labels[idx_val].cpu(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro'))

        # early stopping
        if epoch > early_stopping and loss_val[-1] > np.mean(loss_val[-(early_stopping+1):-1]):
            break

    # test
    with torch.no_grad():
        output = net_gcn(features, adj, val_test=True)
        # print('')
        acc = f1_score(labels[idx_test].cpu(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
        # print('test acc', acc)

    # attack
    w0 = np.load('./weights/' + dataset + '_w0.npy').transpose()
    w1 = np.load('./weights/' + dataset + '_w1.npy').transpose()

    adj_raw, features_raw, labels_raw = load_data_raw(dataset)

    correct_pred_link = 0
    correct_pred_feat = 0
    correct_pred_link_feat = 0
    n_attack = args['nattack']
    for idxt, n in zip(idx_test, range(1000)):

        # link
        pernode = [idxt]
        _, _, adj_per, features_per, _ = graph_attack(adj_raw, features_raw, labels_raw, w0, w1, False, True, pernode, n=n_attack)
        features_per, adj_per = preprocess_feat_adj(features_per, adj_per)
        with torch.no_grad():
            output = net_gcn(features_per, adj_per, val_test=True)[idxt].cpu().numpy().argmax()
            if output == labels[idxt].cpu().numpy():
                correct_pred_link = correct_pred_link + 1
            print(output, labels[idxt].cpu().numpy())
            print(correct_pred_link, n + 1)

        # feat
        pernode = [idxt]
        _, _, adj_per, features_per, _ = graph_attack(adj_raw, features_raw, labels_raw, w0, w1, True, False, pernode, n=n_attack)
        features_per, adj_per = preprocess_feat_adj(features_per, adj_per)
        with torch.no_grad():
            output = net_gcn(features_per, adj_per, val_test=True)[idxt].cpu().numpy().argmax()
            if output == labels[idxt].cpu().numpy():
                correct_pred_feat = correct_pred_feat + 1
            print(output, labels[idxt].cpu().numpy())
            print(correct_pred_feat, n + 1)

        # link & feat
        pernode = [idxt]
        _, _, adj_per, features_per, _ = graph_attack(adj_raw, features_raw, labels_raw, w0, w1, True, True, pernode, n=n_attack)
        features_per, adj_per = preprocess_feat_adj(features_per, adj_per)
        with torch.no_grad():
            output = net_gcn(features_per, adj_per, val_test=True)[idxt].cpu().numpy().argmax()
            if output == labels[idxt].cpu().numpy():
                correct_pred_link_feat = correct_pred_link_feat + 1
            print(output, labels[idxt].cpu().numpy())
            print(correct_pred_link_feat, n + 1)

    adv_acc_link = correct_pred_link / 1000
    adv_acc_feat = correct_pred_feat / 1000
    adv_acc_link_feat = correct_pred_link_feat / 1000

    return acc, adv_acc_link, adv_acc_feat, adv_acc_link_feat


def parser_loader():
    parser = argparse.ArgumentParser(description='Attack GCN')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nattack', type=int, default=2)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    n_repeat = 5
    acc = np.zeros(n_repeat)
    adv_acc_link = np.zeros(n_repeat)
    adv_acc_feat = np.zeros(n_repeat)
    adv_acc_link_feat = np.zeros(n_repeat)
    for seed in range(n_repeat):
        acc[seed], adv_acc_link[seed], adv_acc_feat[seed], adv_acc_link_feat[seed] = run(args, seed)
        # print(acc[seed])

    print('')
    print('clean mean', acc.mean(), 'std', acc.std())
    print('link mean', adv_acc_link.mean(), 'std', adv_acc_link.std())
    print('feat mean', adv_acc_feat.mean(), 'std', adv_acc_feat.std())
    print('link feat mean', adv_acc_link_feat.mean(), 'std', adv_acc_link_feat.std())

