"""Compute clustering meta-features of a dataset.
"""
import math

import numpy as np
from pyclustering.cluster.cure import cure
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from scipy.spatial.distance import squareform

def compute_clustering(meta_features):
    """Compute clustering meta-features of a dataset.

    Args:
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    X = meta_features['utils']['X']
    y = meta_features['utils']['y']

    num_class = meta_features['simple']['num_class']
    num_inst = meta_features['simple']['num_inst']

    dist_p = meta_features['distance']['dist_p']

    num_cluster = num_class

    # Build CURE clustering model
    cure_model = cure(X, num_cluster)
    cure_model.process()

    # Compute basic cluster properties
    clusters = cure_model.get_clusters()
    centroids = cure_model.get_means()
    prop_cluster = []
    compactness = []
    nre = 0
    for idx, cluster in enumerate(clusters):
        prop = len(cluster) / num_inst
        prop_cluster.append(prop)

        cluster_insts = X[cluster]
        cluster_centroid = centroids[idx]
        compact = 0
        for cluster_inst in cluster_insts:
            compact += np.linalg.norm(cluster_inst - cluster_centroid)
        compactness.append(compact)

        nre += prop * math.log(prop / (1 / num_cluster))
    nre = nre / (2 * math.log(num_cluster))

    # Change clustering result representation from idx list separation to idx labelling
    type_repr = cure_model.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    y_cluster = encoder.get_clusters()

    # Compute purity
    purity = []
    for i in range(num_class):
        class_idxs = np.argwhere(y == i).flatten()
        class_clusters = np.take(y_cluster, class_idxs)
        class_num_cluster = len(np.unique(class_clusters))
        purity.append(class_num_cluster / num_cluster)

    # Compute connectivity
    pairwise_distances = squareform(dist_p)
    np.fill_diagonal(pairwise_distances, np.inf)
    nearest_neighbour_idxs = np.argmin(pairwise_distances, axis=1)
    connectivity = 0
    for idx, nearest_neighbour_idx in enumerate(nearest_neighbour_idxs):
        if y_cluster[idx] != y_cluster[nearest_neighbour_idx]:
            # Instance and nearest neighbour instance are not in the same cluster
            connectivity += 1

    # Store clustering-based meta-features in meta-features dictionary
    clustering = {}
    clustering['num_cluster'] = int(num_cluster)
    clustering['prop_cluster'] = np.array(prop_cluster, dtype=np.float32)
    clustering['purity'] = np.array(purity, dtype=np.float32)
    clustering['compactness'] = np.array(compactness, dtype=np.float32)
    clustering['connectivity'] = int(connectivity)
    clustering['nre'] = float(nre)
    meta_features['clustering'] = clustering
