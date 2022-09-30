"""Meta-labels database methods.
"""
import os
import sqlite3

import numpy as np

from .utils import arr_to_text, text_to_arr

CURR_DIR = os.path.dirname(__file__)
META_LABELS_DB_FILE_PATH = os.path.join(CURR_DIR, '..\\db\\meta_labels.db')

def connect_db():
    """Connect to the meta-labels database.

    Returns:
        Connection: Connection to the meta-labels database.
    """
    sqlite3.register_adapter(np.ndarray, arr_to_text)
    sqlite3.register_converter('ARRAY', text_to_arr)
    return sqlite3.connect(META_LABELS_DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)

def init_storage():
    """Initialise the meta-labels database.
    """
    con = connect_db()
    cur = con.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS MetaLabels (
        id          INTEGER PRIMARY KEY,
        accuracy    ARRAY
    );
    ''')
    con.commit()
    con.close()

def store_meta_labels(dataset_id, meta_labels):
    """Store meta-labels of a dataset.

    Args:
        id (int): Dataset ID.
        meta_labels (dict of str: NDArray): Meta-labels dictionary for a dataset with
            an evaluation metric as the key and
            its corresponding evaluation metric scores as the value.
    """
    con = connect_db()
    cur = con.cursor()
    values = [dataset_id]
    for metric_scores in meta_labels.values():
        values.append(metric_scores)
    values = tuple(values)
    cur.execute('INSERT INTO MetaLabels VALUES (?, ?)', values)
    con.commit()
    con.close()

def get_stored_ids():
    """Get IDs of dataset with meta-labels stored in the meta-labels dataset.

    Returns:
        list[int]: List of dataset IDs.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT id FROM MetaLabels')
    ids = []
    for row in cur.fetchall():
        ids.append(*row)
    con.commit()
    con.close()
    return ids

def get_all_meta_labels():
    """Get meta-labels for all datasets.

    Returns:
        NDArray: Array of meta-labels for all datasets with shape (num_datasets, num_metrics, num_models).
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT accuracy FROM MetaLabels')
    meta_labels_rows = cur.fetchall()
    meta_labels = [list(meta_labels_tup) for meta_labels_tup in meta_labels_rows]
    con.commit()
    con.close()
    return np.array(meta_labels)

def delete_meta_labels(dataset_id):
    """Delete meta-labels of a dataset.

    Args:
        dataset_id (int): Dataset ID.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('DELETE FROM MetaLabels WHERE id = ?', (dataset_id,))
    con.commit()
    con.close()
