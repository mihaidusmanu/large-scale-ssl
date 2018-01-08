import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.decomposition import PCA

import anchors_SSL
import fuzzy_cmeans_anchors
import kmeans_anchors
import LAE
import random_anchors
import util

def unpickle(file):
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding = 'bytes')
    return d

def preprocess_cifar10(dir_path):
    x = np.array([])
    labels = np.array([])

    d = unpickle(dir_path + '/data_batch_1')
    x = d[b'data']
    labels = d[b'labels']
    for i in range(2, 6):
        d = unpickle(dir_path + '/data_batch_' + str(i))
        x = np.concatenate((x, d[b'data']))
        labels = np.concatenate((labels, np.array(d[b'labels'])))
    d = unpickle(dir_path + '/test_batch')
    x = np.concatenate((x, d[b'data']))
    labels = np.concatenate((labels, np.array(d[b'labels'])))
 
    n_pixels = 32 * 32
    x = x / 255.
    gray_x = np.zeros((x.shape[0], n_pixels))
    for j in range(n_pixels):
        gray_x[:, j] = 0.21 * x[:, j] + 0.72 * x[:, n_pixels + j] + 0.07 * x[:, 2 * n_pixels + j]
    
    pca = PCA(n_components = 128)
    pca.fit(gray_x)

    final_x = pca.transform(gray_x)
    
    np.save('data/cifar-10_data.npy', final_x)
    np.save('data/cifar-10_labels.npy', labels)
    
K = 10
m = 1000
l = 1000

s = 3

#preprocess_cifar10('data/cifar-10-batches-py')

X = np.load('data/cifar-10_data.npy')
ground_truth = np.load('data/cifar-10_labels.npy')

acc = 0

for t in range(5):
    #anchors, Z = random_anchors.find(X, m, 2, s)
    #anchors, Z = kmeans_anchors.find(X, m, 2, s)
    #anchors, Z = fuzzy_cmeans_anchors.find(X, m, 1.5)
    anchors, Z = LAE.find(X, m, s)
    
    labeled = util.balanced_random_choice(l, K, ground_truth)
    Y = np.zeros((l, K))
    for id, real_id in enumerate(labeled):
        Y[id, ground_truth[real_id]] = 1
    
    pred = anchors_SSL.predict(0.040, Z, labeled, Y)
    
    current_acc = util.accuracy(ground_truth, pred)
    acc += current_acc
    
    print("[Try #" + str(t) + "] " + str(current_acc))

print("Average accuracy over 5 tries: " + str(acc / 5))
