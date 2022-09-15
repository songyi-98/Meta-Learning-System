"""Compute case-based meta-features of a dataset.
"""
import numpy as np
from scipy.spatial.distance import squareform

def compute_case(meta_features):
    """Compute case-based meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    y = meta_features['utils']['y']

    num_inst = meta_features['simple']['num_inst']

    dist_p = meta_features['distance']['dist_p']

    pairwise_distances = squareform(dist_p)
    num_repeated = 0
    num_repeated_diff_labels = 0
    num_no_overlap = 0
    for i in range(1, num_inst):
        for j in range(0, i):
            if pairwise_distances[i][j] == 0:
                num_repeated += 1
                if y[i] != y[j]:
                    num_repeated_diff_labels += 1

    for i in range(num_inst):
        for j in range(i + 1, num_inst):
            if pairwise_distances[i][j] == 0:
                num_repeated += 1
                if y[i] != y[j]:
                    num_repeated_diff_labels += 1

            attrs_diff = X[i] - X[j]
            num_attrs_overlap = np.count_nonzero(attrs_diff == 0)
            if num_attrs_overlap < 2:
                num_no_overlap += 1
    uniqueness = num_repeated / num_inst
    consistency = num_repeated_diff_labels / num_inst
    incoherence = num_no_overlap / num_inst

    # Store case-based meta-features in meta-features dictionary
    case = {}
    case['uniqueness'] = float(uniqueness)
    case['consistency'] = float(consistency)
    case['incoherence'] = float(incoherence)
    meta_features['case'] = case
