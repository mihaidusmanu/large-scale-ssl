import csv
import numpy as np
import matplotlib.pyplot as plt

import anchors_SSL
import fuzzy_cmeans_anchors
import kmeans_anchors
import LAE
import random_anchors
import util

def read_usps(file_path):
    x = []
    labels = []
    f = open(file_path, 'r')
    reader = csv.reader(f, delimiter = ' ')
    for row in reader:
        labels.append(int(float(row[0])))
        x.append([float(v) for v in row[1:-1]])
    f.close()
    return np.array(x), np.array(labels)
    
K = 10
m = 100
l = 100

s = 3

X, ground_truth = read_usps('data/zip.train')

acc = 0

for t in range(5):
    #anchors, Z = random_anchors.find(X, m, 1, s)
    #anchors, Z = kmeans_anchors.find(X, m, 1, s)
    #anchors, Z = fuzzy_cmeans_anchors.find(X, m, 1.10)
    anchors, Z = LAE.find(X, m, s)
    
    labeled = util.random_choice(l, K, ground_truth)
    Y = np.zeros((l, K))
    for id, real_id in enumerate(labeled):
        Y[id, ground_truth[real_id]] = 1
    
    pred = anchors_SSL.predict(0.040, Z, labeled, Y)
    
    current_acc = util.accuracy(ground_truth, pred)
    acc += current_acc
    
    print("[Try #" + str(t) + "] " + str(current_acc))

print("Average accuracy over 5 tries: " + str(acc / 5))
