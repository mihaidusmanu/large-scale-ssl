import numpy as np
import scipy.spatial
from sklearn.cluster import KMeans

eps = 1e-4
max_iter = 300

def find(X, nb_anchors, m):
    N = X.shape[0]
    C = nb_anchors

    # K-Means initialization
    kmeans = KMeans(n_clusters = C).fit(X)
    centers = kmeans.cluster_centers_
    
    # Fuzzy C-Means
    it = 0
    while True:
        inv_sqdist = 1 / scipy.spatial.distance.cdist(X, centers, metric = 'sqeuclidean')
        W = inv_sqdist ** (1 / (m - 1))
        W = W / np.transpose(np.sum(W, 1)[np.newaxis])
        
        old_centers = np.copy(centers)
        
        for c in range(C):
            centers[c, :] = np.dot(W[:, c], X) / np.sum(W[:, c])

        it += 1
        if np.linalg.norm(centers - old_centers) < eps or it > max_iter:
            break

    # Compute final assignment scores
    inv_sqdist = 1 / scipy.spatial.distance.cdist(X, centers, metric = 'sqeuclidean')
    W = inv_sqdist ** (1 / (m - 1))
    W = W / np.transpose(np.sum(W, 1)[np.newaxis])

    return centers, W

if __name__ == '__main__':
    X = np.array([[0,0], [1,0], [0,1], [1,1]])
    print(find(X, 2, 2))
