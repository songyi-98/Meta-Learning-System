"""Compute data distribution meta-features of a dataset.
"""
import numpy as np

def compute_conc(u, v):
    """Compute concentration coefficient of vectors u and v.

    Args:
        u (NDArray): Array of shape (num_inst).
        v (NDArray): Array of shape (num_inst).

    Returns:
        float: Concentration coefficient of vectors u and v.
    """
    u_unique, u = np.unique(u, return_inverse=True)
    u_bins = len(u_unique)
    v_unique, v = np.unique(v, return_inverse=True)
    v_bins = len(v_unique)
    hist, _, _ = np.histogram2d(u, v, [u_bins, v_bins])
    joint_probs = hist / hist.sum()

    pi_ip_arr = []
    for i in range(u_bins):
        pi_ip = 0
        for j in range(v_bins):
            pi_ip += joint_probs[i][j]
        pi_ip_arr.append(pi_ip)

    sum_pi_pj_sq = 0
    for j in range(v_bins):
        pi_pj = 0
        for i in range(u_bins):
            pi_pj += joint_probs[i][j]
        sum_pi_pj_sq += pi_pj**2

    sum_frac = 0
    for i in range(u_bins):
        for j in range(v_bins):
            sum_frac += joint_probs[i][j]**2 / pi_ip_arr[i]

    return (sum_frac - sum_pi_pj_sq) / (1 - sum_pi_pj_sq)

def compute_data_distribution(meta_features):
    """Compute data distribution meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    y = meta_features['utils']['y']
    attrs = meta_features['utils']['attrs']

    num_attr = meta_features['simple']['num_attr']
    num_inst = meta_features['simple']['num_inst']

    cov_eigenvalues = meta_features['statistical']['cov_eigenvalues']

    # Compute sparsity
    sparsity = [(1 / (num_inst - 1)) * (num_inst / len(np.unique(attr)) - 1) for attr in attrs]

    # Compute attributes concentration coefficient
    conc_attr = np.zeros(num_attr * (num_attr - 1))
    index = 0
    for i in range(num_attr):
        for j in range(num_attr):
            if i != j:
                conc_ij = compute_conc(attrs[i], attrs[j])
                conc_attr[index] = conc_ij
                index += 1

    # Compute class concentration coefficient
    conc_class = np.zeros(num_attr)
    for i in range(num_attr):
        conc_iy = compute_conc(attrs[i], y)
        conc_class[i] = conc_iy

    # Compute proportion of PCA explaining specific variance
    var_thr = 0.95 # threshold
    thr_count = 0
    eigenvalues_desc = -np.sort(-cov_eigenvalues)
    size = len(eigenvalues_desc)
    for i in range(size):
        if (i + 1) * eigenvalues_desc[i] > var_thr:
            thr_count += 1
    prop_pca = (size - thr_count + 1) / size

    # Store data distribution meta-features in meta-features dictionary
    data_distribution = {}
    data_distribution['sparsity'] = np.array(sparsity, dtype=np.float32)
    data_distribution['conc_attr'] = np.array(conc_attr, dtype=np.float32)
    data_distribution['conc_class'] = np.array(conc_class, dtype=np.float32)
    data_distribution['prop_pca'] = float(prop_pca)
    meta_features['data_distribution'] = data_distribution
