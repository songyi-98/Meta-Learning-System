"""Dataset user methods.
"""
import logging

from ..system.datasets.pre_processor import pre_process_dataset
from ..system.db.datasets import init_storage, store_dataset, get_dataset_info, delete_datasets

def get_datasets():
    """Get a list of dataset names and sources.

    Returns:
        list[tuple[str, str]]: List of tuples of dataset name and dataset source.
    """
    init_storage()
    return get_dataset_info()

def add_dataset(name, source, col_types, data):
    """Add a dataset by pre-processing and storing it.

    Args:
        name (str): Dataset name.
        source (str): Dataset source.
        col_types (str): Column types in dataset information file.
        data (str): Data in dataset file.

    Raises:
        Exception: If any dataset information provided by the user is invalid.
    """
    try:
        init_storage()
        logging.info('Pre-processing of \'%s\' dataset...', name)
        col_types = col_types.split()
        col_types, dataset_arrs, num_missing_val = pre_process_dataset(name, data, col_types)
        assert len(dataset_arrs) == 1
        logging.info('Completed pre-processing of \'%s\' dataset.', name)
    except:
        logging.exception('\'%s\' dataset cannot be pre-processed.', name)
        raise Exception('Dataset cannot be added. Please check the debug log for details.') from None

    try:
        logging.info('Storing of \'%s\' dataset...', name)
        store_dataset(name, source, col_types, dataset_arrs[0], num_missing_val)
        logging.info('Completed storing of \'%s\' dataset.', name)
    except:
        logging.exception('\'%s\' dataset cannot be stored.', name)
        raise Exception('Dataset cannot be added. Please check the debug log for details.') from None

def delete_dataset(name):
    """Delete a dataset.

    Args:
        name (str): Dataset name.

    Raises:
        Exception: If dataset cannot be deleted.
    """
    try:
        init_storage()
        delete_datasets([name])
    except:
        logging.exception('\'%s\' dataset cannot be deleted.', name)
        raise Exception('Dataset cannot be deleted. Please check the debug log for details.') from None
