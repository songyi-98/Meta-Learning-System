"""Compute landmarking meta-features of a dataset.
"""
import random
from time import perf_counter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def compute_landmarking(meta_features):
    """Compute landmarking meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    y = meta_features['utils']['y']
    attrs = meta_features['utils']['attrs']

    num_attr = meta_features['simple']['num_attr']

    var_importance = meta_features['decision_tree']['var_importance']

    time_start = perf_counter()

    # Get best, worst and random attributes
    attr_idx_best = np.argmax(var_importance)
    attr_idx_worst = np.argmin(var_importance)
    random.seed(0)
    attr_idx_random = random.randint(0, num_attr - 1)

    X_best = np.array([attrs[attr_idx_best]]).T
    X_worst = np.array([attrs[attr_idx_worst]]).T
    X_random = np.array([attrs[attr_idx_random]]).T

    # 5*2-fold stratified CV
    num_cv_iter = 5
    num_fold = 2

    # Evaluate nearest neighbour models
    nn_one_scores = 0
    nn_elite_scores = 0
    for _ in range(num_cv_iter):
        nn_clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        nn_one_scores += np.mean(cross_val_score(nn_clf, X, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1))
        nn_elite_scores += np.mean(
            cross_val_score(nn_clf, X_best, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1)
        )
    nn_one = nn_one_scores / num_cv_iter
    nn_elite = nn_elite_scores / num_cv_iter

    # Evaluate decision tree models
    dt_scores_best = 0
    dt_scores_worst = 0
    dt_scores_random = 0
    for i in range(num_cv_iter):
        dt_clf = DecisionTreeClassifier(random_state=i, class_weight='balanced')
        dt_scores_best += np.mean(
            cross_val_score(dt_clf, X_best, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1)
        )
        dt_scores_worst += np.mean(
            cross_val_score(dt_clf, X_worst, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1)
        )
        dt_scores_random += np.mean(
            cross_val_score(dt_clf, X_random, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1)
        )
    node_best = dt_scores_best / num_cv_iter
    node_worst = dt_scores_worst / num_cv_iter
    node_random = dt_scores_random / num_cv_iter

    # Evaluate linear discriminant analysis model
    ld_scores = 0
    for _ in range(num_cv_iter):
        ld_clf = LinearDiscriminantAnalysis()
        ld_scores += np.mean(cross_val_score(ld_clf, X, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1))
    ld = ld_scores / num_cv_iter

    # Evaluate Naive Bayes model
    nb_scores = 0
    for _ in range(num_cv_iter):
        nb_clf = GaussianNB()
        nb_scores += np.mean(cross_val_score(nb_clf, X, y, scoring='roc_auc_ovo_weighted', cv=num_fold, n_jobs=-1))
    nb = nb_scores / num_cv_iter

    time_end = perf_counter()
    elapsed_time = time_end - time_start

    # Store landmarking meta-features in meta-features dictionary
    landmarking = {}
    landmarking['nn_one'] = nn_one
    landmarking['nn_elite'] = nn_elite
    landmarking['node_best'] = node_best
    landmarking['node_worst'] = node_worst
    landmarking['node_random'] = node_random
    landmarking['ld'] = ld
    landmarking['nb'] = nb
    meta_features['landmarking'] = landmarking

    # Store time-based meta feature in meta-features dictionary
    time = meta_features['time']
    time['time_l'] = elapsed_time
