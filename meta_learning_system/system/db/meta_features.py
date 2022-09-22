"""Meta-features database methods.
"""
import os
import sqlite3

import numpy as np

from .utils import arr_to_text, text_to_arr

CURR_DIR = os.path.dirname(__file__)
META_FEATURES_DB_FILE_PATH = os.path.join(CURR_DIR, '..\\db\\meta_features.db')

def connect_db():
    """Connect to the meta-features database.

    Returns:
        Connection: Connection to the meta-features database.
    """
    sqlite3.register_adapter(np.ndarray, arr_to_text)
    sqlite3.register_converter('ARRAY', text_to_arr)
    return sqlite3.connect(META_FEATURES_DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)

def init_storage():
    """Initialise the meta-features database.
    """
    con = connect_db()
    cur = con.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS MetaFeatures (
        id             INTEGER,
        FOREIGN KEY(id) REFERENCES Datasets(id)
    );

    CREATE TABLE IF NOT EXISTS SimpleMetaFeatures (
        id                  INTEGER,
        num_attr            INTEGER,
        num_class           INTEGER,
        num_inst            INTEGER,
        attr_to_inst        REAL,
        class_to_attr       REAL,
        inst_to_attr        REAL,
        inst_to_class       REAL,
        num_attr_cat        INTEGER,
        num_attr_numeric    INTEGER,
        cat_to_numeric      REAL,
        numeric_to_cat      REAL,
        num_attr_bin        INTEGER,
        prop_class          ARRAY,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS StatisticalMetaFeatures (
        id                  INTEGER,
        max                 ARRAY,
        min                 ARRAY,
        range               ARRAY,
        mean                ARRAY,
        mean_g              ARRAY,
        mean_h              ARRAY,
        mean_t              ARRAY,
        median              ARRAY,
        mad                 ARRAY,
        sd                  ARRAY,
        var                 ARRAY,
        iqr                 ARRAY,
        num_attr_normal     INTEGER,
        num_attr_outlier    INTEGER,
        skewness            ARRAY,
        kurtosis            ARRAY,
        cov                 ARRAY,
        cor                 ARRAY,
        cov_eigenvalues     ARRAY,
        attr_cor            REAL,
        can_cor             ARRAY,
        num_disc            INTEGER,
        wilks_lambda        REAL,
        sd_ratio            REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS InformationTheoreticMetaFeatures (
        id                INTEGER,
        ent_attr          ARRAY,
        ent_class         REAL,
        ent_joint         ARRAY,
        mut_info          ARRAY,
        num_attr_equiv    REAL,
        noisiness         REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS DistanceBasedMetaFeatures (
        id              INTEGER,
        dist_p          ARRAY,
        dist_cp         ARRAY,
        dist_w          ARRAY,
        gravity         REAL,
        cohesiveness    ARRAY,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS DecisionTreeMetaFeatures (
        id                  INTEGER,
        num_leaf            INTEGER,
        num_node            INTEGER,
        node_to_attr        REAL,
        node_to_inst        REAL,
        tree_depth          ARRAY,
        tree_shape          ARRAY,
        tree_imbalance      ARRAY,
        leaves_per_class    ARRAY,
        leaves_branch       ARRAY,
        leaves_corrob       ARRAY,
        leaves_homo         ARRAY,
        nodes_per_level     ARRAY,
        nodes_repeated      ARRAY,
        var_importance      ARRAY,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS ClusteringMetaFeatures (
        id              INTEGER,
        num_cluster     INTEGER,
        prop_cluster    ARRAY,
        purity          ARRAY,
        compactness     ARRAY,
        connectivity    INTEGER,
        nre             REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS LandmarkingMetaFeatures (
        id             INTEGER,
        nn_one         REAL,
        nn_elite       REAL,
        node_best      REAL,
        node_worst     REAL,
        node_random    REAL,
        ld             REAL,
        nb             REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS DataDistributionMetaFeatures (
        id            INTEGER,
        sparsity      ARRAY,
        conc_attr     ARRAY,
        conc_class    ARRAY,
        prop_pca      REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS CaseBasedMetaFeatures (
        id             INTEGER,
        uniqueness     REAL,
        consistency    REAL,
        incoherence    REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS StructuralInformationMetaFeatures (
        id             INTEGER,
        itemset_one    ARRAY,
        itemset_two    ARRAY,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );

    CREATE TABLE IF NOT EXISTS TimeBasedMetaFeatures (
        id         INTEGER,
        time_s     REAL,
        time_it    REAL,
        time_dt    REAL,
        time_l     REAL,
        FOREIGN KEY(id) REFERENCES MetaFeatures(id)
    );
    ''')
    con.commit()
    con.close()

def store_meta_features(dataset_id, meta_features):
    """Store meta-features of a dataset.

    Args:
        id (int): Dataset ID.
        meta_features (dict of str: dict): Meta-features dictionary with
            the dictionary name as the key and
            its corresponding dictionary as the value.
    """
    con = connect_db()
    cur = con.cursor()

    cur.execute('INSERT INTO MetaFeatures VALUES (?)', (dataset_id,))

    simple = meta_features['simple']
    values = [dataset_id]
    for feature in simple.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO SimpleMetaFeatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing simple meta-features.') from exc

    statistical = meta_features['statistical']
    values = [dataset_id]
    for feature in statistical.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO StatisticalMetaFeatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing statistical meta-features.') from exc

    information_theoretic = meta_features['information_theoretic']
    values = [dataset_id]
    for feature in information_theoretic.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO InformationTheoreticMetaFeatures VALUES (?, ?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing information theoretic meta-features.') from exc

    distance = meta_features['distance']
    values = [dataset_id]
    for feature in distance.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO DistanceBasedMetaFeatures VALUES (?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing distance-based meta-features.') from exc

    decision_tree = meta_features['decision_tree']
    values = [dataset_id]
    for feature in decision_tree.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO DecisionTreeMetaFeatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing decision tree meta-features.') from exc

    clustering = meta_features['clustering']
    values = [dataset_id]
    for feature in clustering.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO ClusteringMetaFeatures VALUES (?, ?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing clustering meta-features.') from exc

    landmarking = meta_features['landmarking']
    values = [dataset_id]
    for feature in landmarking.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO LandmarkingMetaFeatures VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing landmarking meta-features.') from exc

    data_distribution = meta_features['data_distribution']
    values = [dataset_id]
    for feature in data_distribution.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO DataDistributionMetaFeatures VALUES (?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing data distribution meta-features.') from exc

    case = meta_features['case']
    values = [dataset_id]
    for feature in case.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO CaseBasedMetaFeatures VALUES (?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing case-based meta-features.') from exc

    structural_information = meta_features['structural_information']
    values = [dataset_id]
    for feature in structural_information.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO StructuralInformationMetaFeatures VALUES (?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing structural information meta-features.') from exc

    time = meta_features['time']
    values = [dataset_id]
    for feature in time.values():
        values.append(feature)
    values = tuple(values)
    try:
        cur.execute(
            'INSERT INTO TimeBasedMetaFeatures VALUES (?, ?, ?, ?, ?)',
            values
        )
    except Exception as exc:
        raise Exception('Error storing time-based meta-features.') from exc

    con.commit()
    con.close()

def get_stored_ids():
    """Get IDs of dataset with meta-features stored in the meta-features dataset.

    Returns:
        list[int]: List of dataset IDs.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT id FROM MetaFeatures')
    ids = []
    for row in cur.fetchall():
        ids.append(*row)
    con.commit()
    con.close()
    return ids

def delete_meta_features(dataset_id):
    """Delete meta-features of a dataset.

    Args:
        dataset_id (int): Dataset ID.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('DELETE FROM MetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM SimpleMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM StatisticalMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM InformationTheoreticMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM DistanceBasedMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM DecisionTreeMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM ClusteringMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM LandmarkingMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM DataDistributionMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM CaseBasedMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM StructuralInformationMetaFeatures WHERE id = ?', (dataset_id,))
    cur.execute('DELETE FROM TimeBasedMetaFeatures WHERE id = ?', (dataset_id,))
    con.commit()
    con.close()
