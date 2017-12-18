import numpy as np
import scipy.spatial
from sklearn.cluster import KMeans

def gaussian_kernel(X, Y, sigma):
    sqdist = scipy.spatial.distance.cdist(X, Y, metric = 'sqeuclidean')
    return np.exp(- 1 / (2 * sigma**2) * sqdist)

def find(X, nb_anchors, sigma):
    kmeans = KMeans(n_clusters = nb_anchors).fit(X)
    anchors = kmeans.cluster_centers_
    Z = gaussian_kernel(X, anchors, sigma)
    Z /= np.transpose(np.sum(Z, 1)[np.newaxis])
    return anchors,Z

if __name__ == '__main__':
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(find(X, 2))
