"""Compute decision tree meta-features of a dataset.
"""
import math
from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier

def compute_decision_tree(meta_features):
    """Compute decision tree meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    y = meta_features['utils']['y']

    num_attr = meta_features['simple']['num_attr']
    num_class = meta_features['simple']['num_class']
    num_inst = meta_features['simple']['num_inst']

    time_start = perf_counter()

    # Build decision tree model
    dt_model = DecisionTreeClassifier(random_state=0, class_weight='balanced')
    dt_model = dt_model.fit(X, y)

    # Compute basic tree properties
    num_leaf = dt_model.get_n_leaves()
    num_node = dt_model.tree_.node_count - num_leaf
    var_importance = dt_model.feature_importances_

    if num_attr == 0:
        node_to_attr = math.nan
    else:
        node_to_attr = num_node / num_attr

    if num_inst == 0:
        node_to_inst = math.nan
    else:
        node_to_inst = num_node / num_inst

    # Compute tree structure properties
    children_left = dt_model.tree_.children_left
    children_right = dt_model.tree_.children_right
    node_features = dt_model.tree_.feature
    node_value_arrs = dt_model.tree_.value
    node_stack = [(0, 0)]
    tree_depth = []
    nodes_per_level = np.zeros(dt_model.get_depth())
    nodes_repeated = np.zeros(num_attr)
    leaf_probs = []
    tree_shape =[]
    leaves_per_class = np.zeros(num_class)
    leaves_corrob = []
    leaves_branch = []
    leaves_homo = []
    while len(node_stack) > 0:
        node_id, node_depth = node_stack.pop()
        tree_depth.append(node_depth)
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            nodes_per_level[node_depth] += 1
            nodes_repeated[node_features[node_id]] += 1

            node_stack.append((children_left[node_id], node_depth + 1))
            node_stack.append((children_right[node_id], node_depth + 1))
        else:
            leaf_prob = 1 / math.pow(2, node_depth)
            leaf_probs.append(leaf_prob)
            leaf_shape = -leaf_prob * math.log2(leaf_prob)
            tree_shape.append(leaf_shape)

            leaf_value_arr = node_value_arrs[node_id]
            leaves_per_class[np.argmax(leaf_value_arr)] += 1
            leaves_corrob.append(np.sum(leaf_value_arr) / num_inst)

            leaves_branch.append(node_depth)
            leaves_homo.append(num_leaf / leaf_shape)
    assert len(tree_depth) == num_leaf + num_node
    nodes_per_level = nodes_per_level[1:] # ignore root node
    assert np.sum(np.array(nodes_per_level)) + 1 == num_node
    assert np.sum(np.array(nodes_repeated)) == num_node
    leaves_per_class /= num_leaf
    assert len(leaves_branch) == num_leaf
    assert len(leaves_homo) == num_leaf

    # Compute tree imbalance
    tree_imbalance = []
    for leaf_prob in leaf_probs:
        leaf_z = leaf_prob * np.count_nonzero(np.array(leaf_probs) == leaf_prob)
        tree_imbalance.append(-leaf_z * math.log2(leaf_z))

    time_end = perf_counter()
    elapsed_time = time_end - time_start

    # Store decision tree-based meta-features in meta-features dictionary
    decision_tree = {}
    decision_tree['num_leaf'] = int(num_leaf)
    decision_tree['num_node'] = int(num_node)
    decision_tree['node_to_attr'] = float(node_to_attr)
    decision_tree['node_to_inst'] = float(node_to_inst)
    decision_tree['tree_depth'] = np.array(tree_depth, dtype=np.int32)
    decision_tree['tree_shape'] = np.array(tree_shape, dtype=np.float32)
    decision_tree['tree_imbalance'] = np.array(tree_imbalance, dtype=np.float32)
    decision_tree['leaves_per_class'] = np.array(leaves_per_class, dtype=np.int32)
    decision_tree['leaves_branch'] = np.array(leaves_branch, dtype=np.int32)
    decision_tree['leaves_corrob'] = np.array(leaves_corrob, dtype=np.float32)
    decision_tree['leaves_homo'] = np.array(leaves_homo, dtype=np.float32)
    decision_tree['nodes_per_level'] = np.array(nodes_per_level, dtype=np.int32)
    decision_tree['nodes_repeated'] = np.array(nodes_repeated, dtype=np.int32)
    decision_tree['var_importance'] = np.array(var_importance, dtype=np.float32)
    meta_features['decision_tree'] = decision_tree

    # Store time-based meta feature in meta-features dictionary
    time = meta_features['time']
    time['time_dt'] = elapsed_time
