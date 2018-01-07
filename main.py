import numpy as np
import matplotlib.pyplot as plt

import anchors_SSL
import fuzzy_cmeans_anchors
import kmeans_anchors
import LAE
import random_anchors
import util

from two_moons import two_moons

K = 2
m = 10
l = 10

s = 3

X, ground_truth = two_moons(1000, 1, 1e-2)

#anchors, Z = random_anchors.find(X, m, 1, s)
#anchors, Z = kmeans_anchors.find(X, m, 1, s)
#anchors, Z = fuzzy_cmeans_anchors.find(X, m, 1.1)
anchors, Z = LAE.find(X, m, s)

labeled = util.random_choice(l, K, ground_truth)
Y = np.zeros((l, K))
for id, real_id in enumerate(labeled):
    Y[id, ground_truth[real_id]] = 1

pred = anchors_SSL.predict(0.040, Z, labeled, Y)

print("Accuracy: " + str(util.accuracy(ground_truth, pred)))
plt.scatter(X[:, 0], X[:, 1], c = pred)
plt.scatter(anchors[:, 0], anchors[:, 1], marker = '+')
plt.show()
