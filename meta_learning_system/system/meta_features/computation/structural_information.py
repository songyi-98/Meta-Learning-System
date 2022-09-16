"""Compute structural information meta-features of a dataset.
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def compute_structural_information(meta_features):
    """Compute structural information meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    attrs = meta_features['utils']['attrs']

    num_attr = meta_features['simple']['num_attr']
    num_inst = meta_features['simple']['num_inst']

    # Perform one-hot encoding of dataset
    dataset_binarised = OneHotEncoder().fit_transform(X).toarray()

    # Compute itemset_one
    itemset_one = np.sum(dataset_binarised, axis=0)
    itemset_one = itemset_one / num_inst

    # Compute range of column indices of binarised dataset for each original attribute
    col_indices_pairs = []
    index_start = 0
    index_end = 0
    attr_unique_value_counts = [len(np.unique(attr)) for attr in attrs]
    for count in attr_unique_value_counts:
        index_end += count
        col_indices_pairs.append((index_start, index_end))
        index_start += count

    # Compute itemset_two
    itemset_two = []
    for i in range(num_attr):
        for j in range(i + 1, num_attr):
            i_start = col_indices_pairs[i][0]
            i_end = col_indices_pairs[i][1]
            j_start = col_indices_pairs[j][0]
            j_end = col_indices_pairs[j][1]
            for col_i in range(i_start, i_end):
                for col_j in range(j_start, j_end):
                    col_i_values = dataset_binarised[:, col_i]
                    col_j_values = dataset_binarised[:, col_j]
                    itemset_two.append(np.count_nonzero(col_i_values != col_j_values) / num_inst)

    # Store structural information meta-features in meta-features dictionary
    structural_information = {}
    structural_information['itemset_one'] = np.array(itemset_one, dtype=np.float32)
    structural_information['itemset_two'] = np.array(itemset_two, dtype=np.float32)
    meta_features['structural_information'] = structural_information
