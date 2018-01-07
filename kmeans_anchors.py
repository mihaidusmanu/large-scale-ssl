import numpy as np
import scipy.spatial
from sklearn.cluster import KMeans, MiniBatchKMeans

def gaussian_kernel(X, Y, sigma):
    sqdist = scipy.spatial.distance.cdist(X, Y, metric = 'sqeuclidean')
    return np.exp(- 1 / (2 * sigma**2) * sqdist)

def find(X, nb_anchors, sigma, s):
    kmeans = MiniBatchKMeans(n_clusters = nb_anchors, batch_size = 500).fit(X)
    #kmeans = KMeans(n_clusters = nb_anchors).fit(X)
    anchors = kmeans.cluster_centers_
    Z_full = gaussian_kernel(X, anchors, sigma)
    Z = np.zeros(Z_full.shape)
    for i in range(Z.shape[0]):
        i_nn = np.argsort(-Z_full[i, :])[0 : s]
        Z[i, i_nn] = Z_full[i, i_nn]
    Z /= np.transpose(np.sum(Z, 1)[np.newaxis])
    return anchors,Z

if __name__ == '__main__':
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(find(X, 2, 1, 1))
