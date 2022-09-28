"""Main module for computing, storing and visualising meta-labels.
"""
import logging
from time import perf_counter

import numpy as np

from ..db.datasets import get_dataset_ids, get_dataset
from .models.main import evaluate_models
from ..db.meta_labels import init_storage, store_meta_labels, get_stored_ids, get_all_meta_labels

def main_user():
    """Main function for meta-labels user application.
    """
    has_error = main()
    if has_error:
        raise Exception()

def main():
    """Main function for meta-labels module.

    Returns:
        bool: True if there is an error when generating meta-labels for any dataset or visualising meta-labels.
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
            logging.info('Retrieving of dataset ID#%i...', dataset_id)
            dataset_name, dataset_arr = get_dataset(dataset_id)
            X = dataset_arr[:, :-1]
            y = dataset_arr[:, -1]
            y = y.astype(np.int32)
            logging.info('Completed retrieving of dataset ID#%i.', dataset_id)
        except:
            has_error = True
            logging.exception('Dataset id#%i cannot be retrieved for meta-labels generation.', dataset_id)
            continue

        try:
            logging.info('Computing meta-labels for \'%s\' dataset...', dataset_name)
            time_start = perf_counter()
            meta_labels = evaluate_models(X, y)
            time_end = perf_counter()
            time_elapsed = time_end - time_start
            logging.info(
                'Completed computing meta-labels for \'%s\' dataset. Time taken: %f s',
                dataset_name,
                time_elapsed
            )
        except:
            has_error = True
            logging.exception('Meta-labels cannot be generated for \'%s\' dataset.', dataset_name)
            continue

        try:
            logging.info('Storing meta-labels for \'%s\' dataset...', dataset_name)
            store_meta_labels(dataset_id, meta_labels)
            logging.info('Completed storing meta-labels for \'%s\' dataset.', dataset_name)
        except:
            has_error = True
            logging.exception('Meta-labels for \'%s\' dataset cannot be stored.', dataset_name)

    try:
        meta_labels_all = get_all_meta_labels() # meta-labels for all datasets
    except:
        has_error = True
        logging.exception('Meta-labels for all datasets cannot be retrieved.')
        return

    return has_error

if __name__ == '__main__':
    main()
