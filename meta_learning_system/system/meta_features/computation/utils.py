"""Utilities for computing meta-features.
"""
import numpy as np

def compute_utils(X, y, col_types):
    """Compute utility values to aid the computation of meta-features.

    Args:
        X (NDArray): Array of attribute vectors with shape (num_inst, num_attr).
        y (NDArray): Array of class labels with shape (num_inst,).
        col_types (NDArray): Array of column types with shape (num_attr + 1,).

    Returns:
        dict of str: Object: Utility dictionary with
            the utility name as the key and
            its corresponding value as the value.
    """
    attrs = X.T # each row is values for ONE attribute
    assert len(attrs) == len(col_types) - 1 # num_cols = num_attr + 1 (target variable)
    attr_types = col_types[:-1]

    classes, class_indices, freq_by_classes = np.unique(y, return_inverse=True, return_counts=True)
    assert np.array_equal(classes, np.array(range(len(classes))))
    assert np.array_equal(y, class_indices) # `y` is an integer encoding of the original target values in the dataset

    # Group instances or attributes by each instance's class label
    insts_by_classes = [X[y == class_label] for class_label in classes]
    attrs_by_classes = [insts_in_class.T for insts_in_class in insts_by_classes]
    assert len(classes) == len(insts_by_classes)
    assert len(classes) == len(attrs_by_classes)

    # Store utilities in utilities dictionary
    utils = {}
    utils['X'] = X
    utils['y'] = y
    utils['attrs'] = attrs
    utils['attr_types'] = attr_types
    utils['classes'] = classes
    utils['freq_by_classes'] = freq_by_classes
    utils['insts_by_classes'] = insts_by_classes
    utils['attrs_by_classes'] = attrs_by_classes
    return utils

def init_meta_features(X, y, col_types):
    """Initialise meta-features dictionary.

    Args:
        X (NDArray): Array of attribute vectors with shape (num_inst, num_attr).
        y (NDArray): Array of class labels with shape (num_inst,).
        col_types (NDArray): Array of column types with shape (num_attr + 1,).

    Returns:
        dict of str: dict: Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    meta_features = {}

    time = {}
    meta_features['time'] = time

    utils = compute_utils(X, y, col_types)
    meta_features['utils'] = utils

    return meta_features
