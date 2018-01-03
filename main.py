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
X, ground_truth = two_moons(1000, 1, 1e-2)

#anchors, Z = random_anchors.find(X, 100, 1)

anchors, Z = kmeans_anchors.find(X, 100, 1)

#anchors, Z = fuzzy_cmeans_anchors.find(X, 10, 1.1)

anchors, Z = LAE.find(X, 100)

num_labeled = 500
labeled = np.random.choice(range(X.shape[0]), num_labeled, replace = False)
Y = np.zeros((num_labeled, K))
for id, real_id in enumerate(labeled):
    Y[id, ground_truth[real_id]] = 1

pred = anchors_SSL.predict(0.040, Z, labeled, Y)

print("Accuracy: " + str(util.accuracy(ground_truth, pred)))
plt.scatter(X[:, 0], X[:, 1], c = pred)
plt.scatter(anchors[:, 0], anchors[:, 1], marker = '+')
plt.show()
