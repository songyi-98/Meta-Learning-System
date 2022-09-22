"""Main module for computing meta-features for a dataset.
"""
from .case import compute_case
from .clustering import compute_clustering
from .data_distribution import compute_data_distribution
from .decision_tree import compute_decision_tree
from .distance import compute_distance
from .information_theoretic import compute_information_theoretic
from .landmarking import compute_landmarking
from .simple import compute_simple
from .statistical import compute_statistical
from .structural_information import compute_structural_information
from .utils import init_meta_features

def compute_meta_features(X, y, col_types):
    """Compute meta-features of a dataset.

    Args:
        X (NDArray): Array of attribute vectors with shape (num_inst, num_attr).
        y (NDArray): Array of class labels with shape (num_inst,).
        col_types (NDArray): Array of column types with shape (num_attr + 1,).

    Returns:
        dict of str: dict: Utility dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    meta_features = init_meta_features(X, y, col_types)

    compute_simple(meta_features)
    compute_statistical(meta_features)
    compute_information_theoretic(meta_features)
    compute_distance(meta_features)
    compute_decision_tree(meta_features)
    compute_clustering(meta_features)
    compute_landmarking(meta_features)
    compute_data_distribution(meta_features)
    compute_case(meta_features)
    compute_structural_information(meta_features)

    return meta_features
