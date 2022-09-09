"""Compute information-theoretic meta-features of a dataset.
"""
import math
from time import perf_counter

import numpy as np
from scipy.stats import entropy

def compute_information_theoretic(meta_features):
    """Compute information-theoretic meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    y = meta_features['utils']['y']
    attrs = meta_features['utils']['attrs']

    num_attr = meta_features['simple']['num_attr']
    num_inst = meta_features['simple']['num_inst']
    prop_class = meta_features['simple']['prop_class']

    time_start = perf_counter()

    # Compute class entropy
    ent_class = entropy(prop_class, base=2)

    ent_attr = []
    ent_joint = []
    mut_info = []
    sum_ent_attr = 0
    sum_mut_info = 0
    for attr in attrs:
        # Compute attribute entropy
        _, attr_counts = np.unique(attr, return_counts=True)
        pk_attr = attr_counts / num_inst
        ent = entropy(pk_attr, base=2)
        ent_attr.append(ent)
        sum_ent_attr += ent

        # Compute joint entropy
        attr_and_labels = list(zip(attr, y))
        tuple_dict = {}
        for t in attr_and_labels:
            tuple_dict[t] = tuple_dict.get(t, 0) + 1
        pk_joint = np.array(list(tuple_dict.values())) / num_inst
        joint = 0
        for p_joint in pk_joint:
            joint += p_joint * math.log2(p_joint)
        ent_joint.append(joint)

        # Compute mutual information
        mutual_info = ent + ent_class - joint
        mut_info.append(mutual_info)
        sum_mut_info += mutual_info

    # Compute equivalent number of attributes
    num_attr_equiv = ent_class / ((1 / num_attr) * sum_mut_info)

    # Compute noisiness of attributes
    noisiness = (sum_ent_attr - sum_mut_info) / sum_mut_info

    time_end = perf_counter()
    elapsed_time = time_end - time_start

    # Store information-theoretic meta-features in meta-features dictionary
    information_theoretic = {}
    information_theoretic['ent_attr'] = np.array(ent_attr, dtype=np.float32)
    information_theoretic['ent_class'] = float(ent_class)
    information_theoretic['ent_joint'] = np.array(ent_joint, dtype=np.float32)
    information_theoretic['mut_info'] = np.array(mut_info, dtype=np.float32)
    information_theoretic['num_attr_equiv'] = float(num_attr_equiv)
    information_theoretic['noisiness'] = float(noisiness)
    meta_features['information_theoretic'] = information_theoretic

    # Store time-based meta feature in meta-features dictionary
    time = meta_features['time']
    time['time_it'] = elapsed_time
