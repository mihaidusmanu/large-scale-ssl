import numpy as np
import sklearn.metrics

def accuracy(ground_truth, labels):
    return sklearn.metrics.accuracy_score(ground_truth, labels)
