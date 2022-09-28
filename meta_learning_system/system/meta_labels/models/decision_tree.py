"""Evaluate a dataset using decision tree.
"""
from functools import partial
import math

import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from skopt import gp_minimize
from skopt.space import Categorical, Integer

# Dictionary to cache hyperparameter values as the key and negated mean accuracy score as the value
CACHE_HYPERPARAMS_SCORES = {}

# 5*2-fold stratified CV
NUM_CV_ITER = 5
NUM_FOLD = 2

def optimise(hyperparams_values, hyperparams_names, X, y):
    """Function to optimise hyperparameters using Bayesian optimisation.

    Args:
        hyperparams_values (list[Any]): List of hyperparameter values.
        hyperparams_names (list[str]): List of hyperparameter names.
        X (NDArray): Array of attribute values with shape (num_inst, num_attr).
        y (NDArray): Array of class labels with shape (num_inst,).

    Returns:
        float: Negated mean accuracy score when evaluating a dataset using decision tree.
    """
    # Retrieve from cache if hyperparameter values already exist
    key = tuple(hyperparams_values)
    if key in CACHE_HYPERPARAMS_SCORES:
        return CACHE_HYPERPARAMS_SCORES.get(key)

    hyperparams = dict(zip(hyperparams_names, hyperparams_values))
    scores = 0
    for i in range(NUM_CV_ITER):
        clf = DecisionTreeClassifier(random_state=i, class_weight='balanced')
        clf.set_params(**hyperparams)
        scores += np.mean(cross_val_score(clf, X, y, scoring='accuracy', cv=NUM_FOLD, n_jobs=-1))
    score = -scores # negation to convert maximisation to minimisation problem

    # Cache value
    CACHE_HYPERPARAMS_SCORES[key] = score

    return score

def evaluate_decision_tree_model(X, y, scoring):
    """Evaluate a dataset using decision tree with evaluation metrics.

    Args:
        X (NDArray): Array of attribute values with shape (num_inst, num_attr).
        y (NDArray): Array of class labels with shape (num_inst,).
        scoring (list[str]): List of evaluation metrics.

    Returns:
        NDArray: Array of evaluation metric scores with shape (len(scoring),)
    """
    # Hyperparameter tuning
    num_training = math.floor(len(y) * 0.9)
    max_depth = math.ceil(math.log(num_training))
    hyperparams_names = ['criterion', 'max_depth']
    hyperparams_space = [
        Categorical(['gini', 'entropy', 'log_loss'], name='criterion'),
        Integer(1, max_depth, name='max_depth')
    ]
    optimisation_function = partial(optimise, hyperparams_names=hyperparams_names, X=X, y=y)
    result = gp_minimize(optimisation_function, hyperparams_space, n_calls=50, random_state=0, n_jobs=-1)
    criterion = result.x[0]
    max_depth = result.x[1]

    # Evaluate decision tree model
    metric_scores = np.zeros(len(scoring))
    for i in range(NUM_CV_ITER):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=i, class_weight='balanced')
        scores = cross_validate(clf, X, y, scoring=scoring, cv=NUM_FOLD, n_jobs=-1)
        for idx, scorer_name in enumerate(scoring):
            key = 'test_' + scorer_name
            metric_scores[idx] += np.mean(scores[key])
    return np.array(metric_scores) / NUM_CV_ITER
