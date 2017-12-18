import numpy as np
from two_moons import two_moons
from random_anchors import random_anchors

def knn(x,U,k):
    dist = np.array([np.linalg.norm(U[i,:] - x) for i in range(U.shape[0])])
    return np.argsort(dist)[0:k]

def simplex_proj(z):
    return z

def LAE(X,U,s):
    n = X.shape[0]
    s = U.shape[1]
    Z = np.zeros((n,s))
    for i in range(n):
        k = 10
        i_nn = knn(X[i,0],U,k)
        g = lambda z: np.linalg.norm(X[i,:] - np.dot(np.transpose(U[i_nn,:]),z))**2/2
        grad_g = lambda z: np.transpose((np.squeeze(np.dot(U[i_nn,:],np.dot(np.transpose(U[i_nn,:]),z))) - np.dot(U[i_nn,:],X[i,:]))[np.newaxis])
        print(U[i_nn,:].shape)
        g_tild = lambda beta,v,z: g(v) + np.dot(np.transpose(grad_g(v)),z-v) + beta * np.linalg.norm(z-v)**2 / 2
        old_z_seq = np.ones((k,1))/s
        z_seq = np.ones((k,1))/s
        old_delta_seq = 0
        delta_seq = 1
        beta_seq = 1
        t = 0
        while True:
            t += 1
            alpha = (old_delta_seq - 1) / delta_seq
            v = z_seq + alpha*(z_seq - old_z_seq)
            j = 0
            while True:
                beta = 2**j * beta_seq
                z = simplex_proj(v - 1/beta * grad_g(v))
                if g(z) <= g_tild(beta,v,z):
                    beta_seq = beta
                    z_seq = z
                    break
                j += 1
            delta_seq = (1 + np.sqrt(1 + 4*delta_seq**2))/2
        Z[i,:] = z_seq
    return Z



if __name__ == "__main__":
    X, Y = two_moons(1000,1,1e-2)
    anchors,Z = random_anchors(X,100)
    LAE(X,anchors,1)
