"""Compute simple meta-features of a dataset.
"""
import math

import numpy as np

def compute_simple(meta_features):
    """Compute simple meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    attrs = meta_features['utils']['attrs']
    attr_types = meta_features['utils']['attr_types']
    classes = meta_features['utils']['classes']
    freq_by_classes = meta_features['utils']['freq_by_classes']

    # Compute basic properties
    dataset_shape = X.shape
    num_inst = dataset_shape[0]
    num_attr = dataset_shape[1]

    # Compute attribute-related properties
    num_attr_cat = np.count_nonzero(attr_types == 'cat')
    num_attr_numeric = np.count_nonzero(attr_types == 'num')
    assert num_attr == num_attr_cat + num_attr_numeric

    num_attr_bin = 0
    for attr in attrs:
        if len(np.unique(attr)) == 2:
            num_attr_bin += 1

    # Compute class-related properties
    num_class = len(classes)
    prop_class = [freq_class / num_inst for freq_class in freq_by_classes]

    # Compute ratios
    if num_attr == 0:
        class_to_attr = math.nan
        inst_to_attr = math.nan
    else:
        class_to_attr = num_class / num_attr
        inst_to_attr = num_inst / num_attr

    if num_class == 0:
        inst_to_class = math.nan
    else:
        inst_to_class = num_inst / num_class

    if num_inst == 0:
        attr_to_inst = math.nan
    else:
        attr_to_inst = num_attr / num_inst

    if num_attr_cat == 0:
        numeric_to_cat = math.nan
    else:
        numeric_to_cat = num_attr_numeric / num_attr_cat

    if num_attr_numeric == 0:
        cat_to_numeric = math.nan
    else:
        cat_to_numeric = num_attr_cat / num_attr_numeric

    # Store simple meta-features in meta-features dictionary
    simple = {}
    simple['num_attr'] = int(num_attr)
    simple['num_class'] = int(num_class)
    simple['num_inst'] = int(num_inst)
    simple['attr_to_inst'] = float(attr_to_inst)
    simple['class_to_attr'] = float(class_to_attr)
    simple['inst_to_attr'] = float(inst_to_attr)
    simple['inst_to_class'] = float(inst_to_class)
    simple['num_attr_cat'] = int(num_attr_cat)
    simple['num_attr_numeric'] = int(num_attr_numeric)
    simple['cat_to_numeric'] = float(cat_to_numeric)
    simple['numeric_to_cat'] = float(numeric_to_cat)
    simple['num_attr_bin'] = int(num_attr_bin)
    simple['prop_class'] = np.array(prop_class, dtype=np.float32)
    meta_features['simple'] = simple
