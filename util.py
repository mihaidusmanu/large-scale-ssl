import numpy as np
import sklearn.metrics

def accuracy(ground_truth, labels):
    return sklearn.metrics.accuracy_score(ground_truth, labels)

def random_choice(l, K, ground_truth):
    N = len(ground_truth)
    labeled = []
    for k in range(K):
        labeled.append(np.random.choice(np.where(ground_truth == k)[0], 1)[0])
    remaining = []
    for i in range(N):
        if i not in labeled:
            remaining.append(i)
    labeled = np.concatenate((np.array(labeled), np.random.choice(remaining, l - K, replace = False)))
    return labeled

def balanced_random_choice(l, K, ground_truth):
    N = len(ground_truth)
    labeled = np.random.choice(np.where(ground_truth == 0)[0], l // K, replace = False)
    for k in range(1, K):
        labeled = np.concatenate((labeled, np.random.choice(np.where(ground_truth == k)[0], l // K, replace = False)))
    return labeled
