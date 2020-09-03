import numpy as np
from sklearn.cluster import KMeans

from utils import load_data


dataset = 'pubmed'
_, features, labels, idx_train, _, _, _ = load_data(dataset)
features = features.numpy()
labels = labels.numpy()

class_num = labels.max() + 1
feat_dim = features.shape[1]

centroids_labeled = np.zeros((class_num, feat_dim))
for cn in range(class_num):
    lf = features[idx_train]
    ll = labels[idx_train]
    centroids_labeled[cn] = lf[ll == cn].mean(axis=0)

cluster_labels = np.ones(labels.shape) * -1
cluster_labels[idx_train] = labels[idx_train]
kmeans = KMeans(n_clusters=200, random_state=0).fit(features)

for cn in range(200):
    centroids_unlabeled = features[kmeans.labels_==cn].mean(axis=0)
    label_for_cluster = np.linalg.norm(centroids_labeled - centroids_unlabeled, axis=1).argmin()
    for node in np.where(kmeans.labels_==cn)[0]:
        if node in idx_train:
            continue
        cluster_labels[node] = label_for_cluster

# print(cluster_labels[:50])
# print(cluster_labels[cluster_labels > 6])
cluster_labels_file = './cluster_labels/' + dataset +'.npy'
np.save(cluster_labels_file, cluster_labels)

