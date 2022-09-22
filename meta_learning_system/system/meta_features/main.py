"""Main module for computing and storing meta-features.
"""
import logging
from time import perf_counter

import numpy as np

from ..db.datasets import get_dataset_ids, get_dataset_col_types, get_dataset
from .computation.main import compute_meta_features
from ..db.meta_features import init_storage, store_meta_features, get_stored_ids, delete_meta_features

def main_user():
    """Main function for meta-features user application.
    """
    has_error = main()
    if has_error:
        raise Exception()

def main():
    """Main function for meta-features module.

    Returns:
        bool: True if there is an error when generating meta-features for any dataset.
    """
    init_storage()

    ids_all = get_dataset_ids()
    ids_stored = get_stored_ids()
    ids_generate = np.setdiff1d(ids_all, ids_stored)
    ids_generate = [31]
    has_error = False
    for dataset_id in ids_generate:
        dataset_id = int(dataset_id)
        try:
            logging.info('Retrieving of dataset ID #%i...', dataset_id)
            dataset_name, dataset_arr = get_dataset(dataset_id)
            X = dataset_arr[:, :-1]
            y = dataset_arr[:, -1]
            y = y.astype(np.int32)
            col_types = get_dataset_col_types(dataset_id)
            logging.info('Completed retrieving of dataset ID #%i: %s.', dataset_id, dataset_name)
        except:
            has_error = True
            logging.exception('Dataset id#%i cannot be retrieved for meta-labels generation.', dataset_id)
            continue

        try:
            logging.info('Computing meta-features for \'%s\' dataset...', dataset_name)
            time_start = perf_counter()
            meta_features = compute_meta_features(X, y, col_types)
            time_end = perf_counter()
            time_elapsed = time_end - time_start
            logging.info(
                'Completed computing meta-features for \'%s\' dataset.  Time taken: %f s',
                dataset_name,
                time_elapsed
            )
        except:
            has_error = True
            delete_meta_features(dataset_id)
            logging.exception('Meta-features cannot be generated for \'%s\' dataset.', dataset_name)
            continue

        try:
            logging.info('Storing meta-features for \'%s\' dataset...', dataset_name)
            store_meta_features(dataset_id, meta_features)
            logging.info('Completed storing meta-features for \'%s\' dataset.', dataset_name)
        except:
            has_error = True
            logging.exception('Meta-features for \'%s\' dataset cannot be stored.', dataset_name)
    return has_error

if __name__ == '__main__':
    main()
