"""Compute distance-based meta-features of a dataset.
"""
import math

import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_distance(meta_features):
    """Compute distance-based meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    y = meta_features['utils']['y']
    freq_by_classes = meta_features['utils']['freq_by_classes']
    insts_by_classes = meta_features['utils']['insts_by_classes']

    num_inst = meta_features['simple']['num_inst']

    # Compute pairwise distances
    dist_p = pdist(X, 'euclidean')
    pairwise_distances = squareform(dist_p)

    # Compute distance and correlations of all pairs of instances
    sd = np.std(X, axis=1)
    cov_insts = np.cov(X)
    cor = []
    for i in range(num_inst):
        for j in range(i + 1, num_inst):
            cov_ij = cov_insts[i][j]
            cor_ij = cov_ij / (sd[i] * sd[j])
            cor.append(cor_ij)
    cor = np.array(cor)
    c_prime = (cor + 1) / 2

    dist_p_min = np.amin(dist_p)
    dist_p_max = np.amax(dist_p)
    d_prime = (dist_p - dist_p_min) / (dist_p_max - dist_p_min)

    dist_cp = [c_prime, d_prime]

    # Compute weighted distance
    dist_w = []
    for i in range(num_inst):
        numerator = 0
        denominator = 0
        for j in range(num_inst):
            if i == j:
                continue
            dist_ij = pairwise_distances[i][j]
            d_ij = dist_ij / math.sqrt(num_inst - dist_ij)
            W_ij = 1 / 2**(2 * d_ij)
            numerator += W_ij * dist_ij
            denominator += W_ij
        dist_w.append(numerator / denominator)

    # Compute centre of gravity
    class_min = np.argmin(freq_by_classes)
    class_max = np.argmax(freq_by_classes)
    X_class_min = insts_by_classes[class_min]
    X_class_max = insts_by_classes[class_max]
    x_class_min = np.mean(X_class_min, axis=0)
    x_class_max = np.mean(X_class_max, axis=0)
    gravity = float(np.linalg.norm(x_class_max - x_class_min))

    # Compute cohesiveness
    cohesiveness = []
    for i in range(num_inst):
        inst_label = y[i]
        class_freq = freq_by_classes[inst_label]
        dists = pairwise_distances[i, :]
        idx_partition = np.argpartition(dists, (1, class_freq))
        k_nearest_classes = y[idx_partition[1:class_freq]] # index 0 is current instance
        c = np.count_nonzero(k_nearest_classes != inst_label)
        c /= class_freq - 1
        cohesiveness.append(c)

    # Store distance-based meta-features in meta-features dictionary
    distance = {}
    distance['dist_p'] = np.array(dist_p, dtype=np.float32)
    distance['dist_cp'] = np.array(dist_cp, dtype=np.float32)
    distance['dist_w'] = np.array(dist_w, dtype=np.float32)
    distance['gravity'] = float(gravity)
    distance['cohesiveness'] = np.array(cohesiveness, dtype=np.float32)
    meta_features['distance'] = distance
