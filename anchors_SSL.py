import numpy as np

def predict(gamma, Z, labeled, Y):
    tZ_Z = np.dot(np.transpose(Z), Z)
    delta = np.sum(Z, 0)
    L = tZ_Z - np.dot(tZ_Z, np.dot(np.diag(1 / delta), tZ_Z))
    
    Z_l = Z[labeled, :]
    
    A = np.dot(np.linalg.inv(np.dot(np.transpose(Z_l), Z_l) + gamma * L), np.dot(np.transpose(Z_l), Y))
    print(A)

    N = Z.shape[0]
    K = Y.shape[1]
    pred = np.zeros((N, K))

    for k in range(K):
        current_pred = np.dot(Z, A[:, k])
        pred[:, k] = current_pred / np.sum(current_pred)
    
    pred[labeled, :] = Y
    
    print(pred)

    return np.argmax(pred, axis=1)
