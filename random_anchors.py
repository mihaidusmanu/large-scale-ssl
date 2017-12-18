import numpy as np
import scipy.spatial

def gaussian_kernel(X,Y,sigma):
    dist = scipy.spatial.distance.cdist(X,Y,metric='sqeuclidean')
    return np.exp(-1/(2*sigma**2) * dist)

def kernel_weights(X,anchors,ker):
    Z = ker(X,anchors)
    return np.array([l / sum(l) for l in Z])

def random_anchors(X,nb_anchors):
    anchors = [X[i,:] for i in np.random.choice(range(nb_anchors),nb_anchors,replace=False)]
    Z = kernel_weights(X,anchors,lambda X,Y: gaussian_kernel(X,Y,1))
    return anchors,Z

if __name__ == '__main__':
    X = np.array([[0, 0], [1,0], [0,1], [1,1]])
    print(random_anchors(X,2))
