"""Main module for generating meta-labels for a dataset.
"""
import logging
from time import perf_counter

import numpy as np

from .decision_tree import evaluate_decision_tree_model
from .knn import evaluate_knn_model

MODELS = ['kNN', 'decision tree']
SCORING = ['accuracy']

def evaluate_models(X, y):
    """Evaluate machine learning models on a dataset.

    Args:
        X (NDArray): Array of attribute values with shape (num_inst, num_attr).
        y (NDArray): Array of class labels with shape (num_inst,).

    Returns:
        dict of str: NDArray: Meta-labels dictionary with
            an evaluation metric as the key and
            its corresponding evaluation metric scores as the value.
    """
    time_start = perf_counter()
    scores_knn = evaluate_knn_model(X, y, SCORING)
    scores_decision_tree = evaluate_decision_tree_model(X, y, SCORING)
    time_end = perf_counter()
    scores_metrics = np.vstack((scores_knn, scores_decision_tree)).T

    meta_labels = {}
    for idx, scores in enumerate(scores_metrics):
        # Scores for different models under a particular evaluation metric
        meta_labels[SCORING[idx]] = np.array(scores, dtype=np.float32)

    time_elapsed = time_end - time_start
    logging.info('Time taken: %f s', time_elapsed)

    return meta_labels
