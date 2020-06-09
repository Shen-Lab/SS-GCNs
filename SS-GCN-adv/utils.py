import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import torch
import metis

from nettack.nettack import utils # , GCN
from nettack.nettack import nettack as ntk 


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # preprocess adj
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = torch.tensor((labels * range(l_num)).sum(axis=1), dtype=torch.int64)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    '''
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    '''

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #return sparse_to_tuple(adj_normalized)
    return adj_normalized


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def graph_attack(Alink, Xfeature, Labels, W1, W2, perturb_features, perturb_structure, Pernode, n=1):
    Degrees = Alink.sum(axis=0)
    x_Degrees = Xfeature.sum(axis=1)
    # Pernode is a list of nodes need to be perturbed
    direct_attack = True
    n_influencers = 1 if direct_attack else 5
    per_feature_dict = {}
    per_structure_dict = {}
    tmpAlink = Alink
    tmpXfeature = Xfeature
    for node in Pernode:
        # n_perturbations = int(Degrees[0, node]) + 2  # How many perturb ations to perform. Default: Degree of the node
        n_perturbations_link = int(Degrees[0, node] / 2 + 1) # How many perturb ations to perform. Default: Degree of the node
        '''
        if n_perturbations_link > 5:
            n_perturbations_link = 5
        '''
        # n_perturbations_node = 20
        # n_perturbations_node = int(Xfeature.shape[1] / 100)
        n_perturbations_node = int(Degrees[0, node] / 2 + 1)

        n_perturbations_link = n
        n_perturbations_node = n * 10

        # link
        if perturb_structure:
            nettack = ntk.Nettack(tmpAlink, tmpXfeature, Labels, W1 , W2, node, verbose=True)
            nettack.reset()
            nettack.attack_surrogate(n_perturbations_link, perturb_structure=True, perturb_features=False, direct=direct_attack, n_influencers=n_influencers)
            # nettack.attack_surrogate(n_perturbations, perturb_structure=True, perturb_features=True, direct=direct_attack, n_influencers=n_influencers)
            tmpAlink = nettack.adj
            tmpXfeature = nettack.X_obs.tocsr()
            per_structure_dict[node] = nettack.structure_perturbations

        if perturb_features:
            # node
            nettack = ntk.Nettack(tmpAlink, tmpXfeature, Labels, W1 , W2, node, verbose=True)
            nettack.reset()
            nettack.attack_surrogate(n_perturbations_node, perturb_structure=False, perturb_features=True, direct=direct_attack, n_influencers=n_influencers)
            tmpAlink = nettack.adj
            tmpXfeature = nettack.X_obs.tocsr()
            per_feature_dict[node] = nettack.feature_perturbations

    # tmpAlink = nettack.adj_preprocessed
    print(per_structure_dict, per_feature_dict)

    return per_structure_dict, per_feature_dict, tmpAlink, tmpXfeature, nettack


def load_data_raw(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    '''
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    '''

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    '''
    # preprocess adj
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    '''

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = (labels * range(l_num)).sum(axis=1)
    labels = labels.astype(np.int64)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels


def preprocess_feat_adj(features, adj):

    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features.cuda(), adj.cuda()

def partition(adj, nparts):

    adj_coo = adj.tocoo()
    node_num = adj_coo.shape[0]
    adj_list = [[] for _ in range(node_num)]
    for i, j in zip(adj_coo.row, adj_coo.col):
        if i == j:
            continue
        adj_list[i].append(j)

    _, partition_labels =  metis.part_graph(adj_list, nparts=nparts, seed=0)
    partition_labels = torch.tensor(partition_labels, dtype=torch.int64)

    return partition_labels.cuda()
