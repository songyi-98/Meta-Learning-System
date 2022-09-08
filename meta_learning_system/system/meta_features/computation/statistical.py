"""Compute statistical meta-features of a dataset.
"""
import math
from time import perf_counter

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder

def compute_statistical(meta_features):
    """Compute statistical meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    y = meta_features['utils']['y']
    attrs = meta_features['utils']['attrs']
    freq_by_classes = meta_features['utils']['freq_by_classes']
    attrs_by_classes = meta_features['utils']['attrs_by_classes']

    num_attr = meta_features['simple']['num_attr']
    num_class = meta_features['simple']['num_class']
    num_inst = meta_features['simple']['num_inst']

    time_start = perf_counter()

    # Compute extrema finding statistics
    mx = np.amax(attrs, axis=1)
    mn = np.amin(attrs, axis=1)
    rg = mx - mn

    # Compute order statistics
    Q1 = np.percentile(attrs, 25, axis=1)
    Q3 = np.percentile(attrs, 75, axis=1)
    iqr = Q3 - Q1

    # Compute summary statistics
    mean = np.mean(attrs, axis=1)
    mean_g = stats.gmean(attrs, axis=1)
    mean_h = stats.hmean(attrs, axis=1)
    mean_t = stats.trim_mean(attrs, 0.2, axis=1)
    median = np.median(attrs, axis=1)
    mad = stats.median_abs_deviation(attrs, axis=1)
    sd = np.std(attrs, axis=1)
    var = np.var(attrs, axis=1)
    skewness = stats.skew(attrs, axis=1)
    kurtosis = stats.kurtosis(attrs, axis=1)

    # Conduct statistical tests
    num_attr_normal = 0
    num_attr_outlier = 0
    for i, attr in enumerate(attrs):
        if stats.shapiro(attr).pvalue > 0.05:
            num_attr_normal += 1

        lower_limit = Q1[i] - 1.5 * iqr[i]
        upper_limit = Q3[i] + 1.5 * iqr[i]
        if np.any(attr < lower_limit) or np.any(attr > upper_limit):
            num_attr_outlier += 1

    # Compute attribute correlation statistics
    cov_attrs = np.cov(attrs)
    cor_attrs = np.corrcoef(attrs)
    cov_eigenvalues = np.linalg.eigvals(cov_attrs)
    cov = []
    cor = []
    attr_cor = 0
    for i in range(num_attr):
        for j in range(i + 1, num_attr):
            cov_ij = cov_attrs[i][j]
            cor_ij = cor_attrs[i][j]
            if cor_ij >= 0.5:
                attr_cor += 1
            cov.append(abs(cov_ij))
            cor.append(abs(cor_ij))
    attr_cor *= 2 / (num_attr * (num_attr - 1))

    # Perform one-hot encoding of class labels
    Y = y.reshape(num_inst, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y)

    # Compute correlation statistics between attribute and class labels
    num_components = min(num_attr, num_class)
    cca = CCA(n_components=num_components)
    X_c, y_c = cca.fit_transform(X, Y)
    can_cor = np.array([np.corrcoef(X_c[:, i].T, y_c[:, i].T)[0, 1] for i in range(num_components)])

    num_disc = len(can_cor)
    assert num_disc == num_components

    wilks_lambda = 1
    for i in range(num_disc):
        can_cor_i = can_cor[i]
        eigenvalue_i = can_cor_i**2 / (1 - can_cor_i**2)
        wilks_lambda *= 1 / (1 + eigenvalue_i)

    # Compute homogeneity of covariances
    sd_ratio = 0
    gamma = 0
    pooled_cov = np.zeros((num_attr, num_attr))
    cov_classes = []
    for i in range(num_class):
        freq_class = freq_by_classes[i]
        attrs_class = attrs_by_classes[i]

        gamma += 1 / (freq_class - 1)
        cov_class = np.cov(attrs_class)
        cov_classes.append(cov_class)
        pooled_cov += (freq_class - 1) * cov_class
    gamma = (
        1 -
        ((2 * num_attr**2 + 3 * num_attr - 1) / (6 * (num_attr + 1) * (num_class - 1))) * gamma -
        (1 / (num_inst - num_class))
    )
    pooled_cov *= 1 / (num_inst - num_class)
    numerator = 0
    denominator = 0
    for i in range(num_class):
        freq_class = freq_by_classes[i]
        cov_class = cov_classes[i]

        try:
            det = np.linalg.det(np.linalg.inv(cov_class) @ pooled_cov)
        except:
            sd_ratio = np.nan
            break
        numerator += (freq_class - 1) * math.log(det)
        denominator += freq_class - 1
    if sd_ratio == 0:
        numerator *= gamma
        denominator *= num_attr
        sd_ratio = math.exp(numerator / denominator)

    time_end = perf_counter()
    elapsed_time = time_end - time_start

    # Store statistical meta-features in meta-features dictionary
    statistical = {}
    statistical['max'] = np.array(mx, dtype=np.float32)
    statistical['min'] = np.array(mn, dtype=np.float32)
    statistical['range'] = np.array(rg, dtype=np.float32)
    statistical['mean'] = np.array(mean, dtype=np.float32)
    statistical['mean_g'] = np.array(mean_g, dtype=np.float32)
    statistical['mean_h'] = np.array(mean_h, dtype=np.float32)
    statistical['mean_t'] = np.array(mean_t, dtype=np.float32)
    statistical['median'] = np.array(median, dtype=np.float32)
    statistical['mad'] = np.array(mad, dtype=np.float32)
    statistical['sd'] = np.array(sd, dtype=np.float32)
    statistical['var'] = np.array(var, dtype=np.float32)
    statistical['iqr'] = np.array(iqr, dtype=np.float32)
    statistical['num_attr_normal'] = int(num_attr_normal)
    statistical['num_attr_outlier'] = int(num_attr_outlier)
    statistical['skewness'] = np.array(skewness, dtype=np.float32)
    statistical['kurtosis'] = np.array(kurtosis, dtype=np.float32)
    statistical['cov'] = np.array(cov, dtype=np.float32)
    statistical['cor'] = np.array(cor, dtype=np.float32)
    statistical['cov_eigenvalues'] = np.array(cov_eigenvalues, dtype=np.float32)
    statistical['attr_cor'] = float(attr_cor)
    statistical['can_cor'] = np.array(can_cor, dtype=np.float32)
    statistical['num_disc'] = int(num_disc)
    statistical['wilks_lambda'] = float(wilks_lambda)
    statistical['sd_ratio'] = float(sd_ratio)
    meta_features['statistical'] = statistical

    # Store time-based meta feature in meta-features dictionary
    time = meta_features['time']
    time['time_s'] = elapsed_time
