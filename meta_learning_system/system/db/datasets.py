"""Datasets database methods.
"""
import os
import sqlite3

import numpy as np

from .utils import arr_to_text, text_to_arr
from .meta_features import delete_meta_features
from .meta_labels import delete_meta_labels

CURR_DIR = os.path.dirname(__file__)
DATASETS_DB_FILE_PATH = os.path.join(CURR_DIR, '..\\db\\datasets.db')

def connect_db():
    """Connect to the datasets database.

    Returns:
        Connection: Connection to the datasets database.
    """
    sqlite3.register_adapter(np.ndarray, arr_to_text)
    sqlite3.register_converter('ARRAY', text_to_arr)
    return sqlite3.connect(DATASETS_DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)

def init_storage():
    """Initialise the datasets database.
    """
    con = connect_db()
    cur = con.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS Datasets (
        id                 INTEGER PRIMARY KEY,
        name               TEXT UNIQUE,
        source             TEXT,
        col_types          ARRAY,
        data               ARRAY,
        num_missing_val    INTEGER
    );
    ''')
    con.commit()
    con.close()

def store_dataset(dataset_name, source, col_types, dataset_arr, num_missing_val):
    """Store a dataset's information in the datasets database.

    Args:
        dataset_name (str): Dataset name.
        source (str): Dataset source.
        col_types (NDArray): Array of column types with shape (num_attr + 1,).
        dataset_arr (NDArray): Dataset array with shape (num_inst, num_attr + 1).
        num_missing_val (int): Number of missing attribute values in a dataset.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute(
        'INSERT INTO Datasets (name, source, col_types, data, num_missing_val) VALUES (?, ?, ?, ?, ?)',
        (dataset_name, source, col_types, dataset_arr, num_missing_val)
    )
    con.commit()
    con.close()

def get_dataset_ids():
    """Get a list of dataset IDs.

    Returns:
        list[int]: List of dataset IDs.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT id FROM Datasets')
    ids = []
    for row in cur.fetchall():
        ids.append(*row)
    con.commit()
    con.close()
    return ids

def get_dataset_info():
    """Get a list of dataset names and sources.

    Returns:
        list[tuple[str, str]]: List of tuples of dataset name and dataset source.
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT name, source FROM Datasets')
    info = []
    for row in cur.fetchall():
        info.append(row)
    con.commit()
    con.close()
    return info

def get_dataset_col_types(dataset_id):
    """Get a dataset's column types based on its ID.

    Args:
        dataset_id (int): Dataset ID.

    Returns:
        NDArray: Array of dataset column types with shape (num_attr + 1,).
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT col_types FROM Datasets WHERE id = ?', (dataset_id,))
    col_types = cur.fetchone()[0]
    con.commit()
    con.close()
    return col_types

def get_dataset(dataset_id):
    """Get a dataset's name and array based on its ID.

    Args:
        dataset_id (int): Dataset ID.

    Returns:
        tuple[str, NDArray]: Tuple of dataset name and dataset array with shape (num_inst, num_attr + 1).
    """
    con = connect_db()
    cur = con.cursor()
    cur.execute('SELECT name, data FROM Datasets WHERE id = ?', (dataset_id,))
    dataset_name, dataset_arr = cur.fetchone()
    con.commit()
    con.close()
    return dataset_name, dataset_arr

def delete_datasets(dataset_names):
    """Delete datasets.

    Args:
        dataset_names (list[str]): List of dataset names.
    """
    con = connect_db()
    cur = con.cursor()
    for name in dataset_names:
        cur.execute('SELECT id FROM Datasets WHERE name = ?', (name,))
        dataset_id = cur.fetchone()[0]
        cur.execute('DELETE FROM Datasets WHERE name = ?', (name,))
        delete_meta_features(dataset_id)
        delete_meta_labels(dataset_id)
    con.commit()
    con.close()
