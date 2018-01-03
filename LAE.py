import numpy as np
import kmeans_anchors

from two_moons import two_moons

def knn(x,U,k):
    dist = np.array([np.linalg.norm(U[i,:] - x) for i in range(U.shape[0])])
    return np.argsort(dist)[0:k]

def simplex_proj(z):
    s = z.shape[0]
    v = np.sort(z,axis=0)[::-1]
    l = [v[j] - 1/(j+1) * (sum(v[:j+1])-1) for j in range(s)]
    rho = max(j if l[j] > 0 else 0 for j in range(s))
    theta = 1/(rho+1) * (sum(v[:rho+1]) - 1)
    return np.maximum(z-theta,np.zeros(z.shape))

def LAE(X,U):
    n = X.shape[0]
    d = U.shape[1]
    s = 10
    Z = np.zeros((n,U.shape[0],1))
    eps = 1e-3
    for i in range(n):
        i_nn = knn(X[i,0],U,s)
        g = lambda z: np.linalg.norm(X[i,:] - np.dot(np.transpose(U[i_nn,:]),z))**2/2
        grad_g = lambda z: np.transpose((np.squeeze(np.dot(U[i_nn,:],np.dot(np.transpose(U[i_nn,:]),z))) - np.dot(U[i_nn,:],X[i,:]))[np.newaxis])
        g_tild = lambda beta,v,z: g(v) + np.dot(np.transpose(grad_g(v)),z-v) + beta * np.linalg.norm(z-v)**2 / 2
        old_z_seq = np.ones((s,1))/s
        z_seq = np.ones((s,1))/s
        old_delta_seq = 0
        delta_seq = 1
        beta_seq = 1
        t = 0
        while t == 0 or np.linalg.norm(old_z_seq - z_seq) > eps:
            t += 1
            alpha = (old_delta_seq - 1) / delta_seq
            v = z_seq + alpha*(z_seq - old_z_seq)
            # print(g(z_seq))
            j = 0
            while True:
                beta = 2**j * beta_seq
                z = simplex_proj(v - 1/beta * grad_g(v))
                # z = (v - 1/beta * grad_g(v))
                if g(z) <= g_tild(beta,v,z):
                    beta_seq = beta
                    old_z_seq = z_seq
                    z_seq = z
                    break
                j += 1
            old_delta_seq = delta_seq
            delta_seq = (1 + np.sqrt(1 + 4*delta_seq**2))/2
        Z[i,i_nn] = z_seq
    return Z

def find(X, nb_anchors):
  anchors, _ = kmeans_anchors.find(X, nb_anchors, 1)
  Z = LAE(X, anchors)
  return anchors, np.squeeze(Z)
