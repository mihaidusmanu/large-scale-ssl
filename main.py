import numpy as np
import matplotlib.pyplot as plt

from two_moons import two_moons
from random_anchors import random_anchors
import anchors_SSL
import util

K = 2
X, ground_truth = two_moons(1000, 1, 1e-2)
anchors, Z = random_anchors(X, 100)
num_labeled = 10
labeled = np.random.choice(range(X.shape[0]), num_labeled, replace = False)
Y = np.zeros((num_labeled, K))
for id, real_id in enumerate(labeled):
    Y[id, ground_truth[real_id]] = 1
pred = anchors_SSL.predict(0.040, Z, labeled, Y)
plt.scatter(X[:, 0], X[:, 1], c = pred)
plt.show()
print(util.accuracy(ground_truth, pred))
