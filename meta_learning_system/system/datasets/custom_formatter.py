"""Custom formatter for a problematic dataset.
"""
import os

CURR_DIR = os.path.dirname(__file__)
RAW_DATASETS_FOLDER_PATH = os.path.join(CURR_DIR, 'datasets_raw\\')
CUSTOM_FILE_NAME = 'custom_file.data'
CUSTOM_FILE_PATH = RAW_DATASETS_FOLDER_PATH + CUSTOM_FILE_NAME

def custom_format(dataset_file_names):
    """Apply custom formatting on a dataset's file(s).

    Args:
        dataset_file_names (list[str]): List of file name(s) for a dataset.

    Raises:
        Exception: If there is no custom formatter for a dataset's file(s).
    """
    formatted_rows = []

    for dataset_file_name in dataset_file_names:
        dataset_file_path = RAW_DATASETS_FOLDER_PATH + dataset_file_name
        with open(dataset_file_path, 'r', encoding='utf-8') as dataset_file:
            rows = dataset_file.read().strip().split('\n')

        if (dataset_file_name == 'allbp.data' or dataset_file_name == 'allbp.test' or
            dataset_file_name == 'allhyper.data' or dataset_file_name == 'allhyper.test' or
            dataset_file_name == 'allhypo.data' or dataset_file_name == 'allhypo.test' or
            dataset_file_name == 'allrep.data' or dataset_file_name == 'allrep.test' or
            dataset_file_name == 'dis.data' or dataset_file_name == 'dis.test' or
            dataset_file_name == 'sick.data' or dataset_file_name == 'sick.test'):
            for row in rows:
                formatted_rows.append(row.split('|')[0])
        else:
            raise Exception(f'\'{dataset_file_name}\' dataset has no custom formatter.')

    write_data = '\n'.join(formatted_rows)
    with open(CUSTOM_FILE_PATH, 'w', encoding='utf-8') as custom_file:
        custom_file.write(write_data)
