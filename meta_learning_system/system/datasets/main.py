"""Main module for pre-processing and storing datasets.
"""
import logging
import os

from .custom_formatter import custom_format
from .pre_processor import read_dataset, pre_process_dataset
from ..db.datasets import init_storage, store_dataset, delete_datasets

CURR_DIR = os.path.dirname(__file__)
FORMATTING_INFO_FILE_PATH = os.path.join(CURR_DIR, 'formatting_information.csv')
CUSTOM_FILE_NAME = 'custom_file.data'

def main():
    """Main function for datasets module.
    """
    init_storage()

    # Read the formatting information CSV file containing steps required to format each dataset.
    #
    # The file contains the following columns:
    # Column 0: Boolean indicating if a dataset is already stored in the datasets database.
    # Column 1: Boolean indicating if a dataset is problematic and requires custom formatting first.
    # Column 2: Dataset name.
    # Column 3: Dataset file name(s) separated by spaces.
    # Column 4: Column types separated by spaces. A column type can be:
    #     'del' for delete column
    #     'cat' for categorical attribute
    #     'num' for numerical attribute
    #     'target' for target variable
    # Column 5: Boolean indicating if commas are used to separate values.
    # Column 6: Dataset source.
    with open(FORMATTING_INFO_FILE_PATH, 'r', encoding='utf-8') as formatting_info_file:
        formatting_info = formatting_info_file.read().strip().split('\n')

    write_rows = []
    write_rows.append(formatting_info[0])

    for dataset_formatting_info in formatting_info[1:]: # column 0 is header
        columns = dataset_formatting_info.split(',')

        is_stored = int(columns[0])
        if is_stored: # dataset is already stored in the datasets database
            write_rows.append(','.join(columns))
            continue

        is_problematic = int(columns[1])
        dataset_name = columns[2]
        dataset_file_names = columns[3].split()
        if is_problematic:
            try:
                logging.info('Custom formatting of \'%s\' dataset...', dataset_name)
                custom_format(dataset_file_names)
                dataset_file_names = [CUSTOM_FILE_NAME]
                logging.info('Completed custom formatting of \'%s\' dataset.', dataset_name)
            except:
                logging.exception('\'%s\' dataset cannot be custom formatted.', dataset_name)
                write_rows.append(','.join(columns))
                continue

        col_types = columns[4].split()
        is_comma_separator = int(columns[5])
        try:
            logging.info('Pre-processing of \'%s\' dataset...', dataset_name)
            dataset = read_dataset(dataset_file_names)
            col_types, dataset_arrs, num_missing_val = pre_process_dataset(
                dataset_name,
                dataset,
                col_types,
                is_comma_separator
            )
            logging.info('Completed pre-processing of \'%s\' dataset.', dataset_name)
        except:
            logging.exception('\'%s\' dataset cannot be pre-processed.', dataset_name)
            write_rows.append(','.join(columns))
            continue

        # TODO: Skip dataset with missing values for now
        if len(dataset_arrs) == 0:
            logging.warning('\'%s\' dataset is skipped due to missing values.', dataset_name)
            write_rows.append(','.join(columns))
            continue

        # If an original unprocessed dataset has x > 1 target variables, `dataset_arrs` will have x arrays where
        # each array has a unique (1 of the x) target variable but all arrays have the same attribute values.
        # In this case, an exception may be raised when storing one of the x arrays.
        # This requires already stored arrays associated with the dataset to be deleted to prevent partial storage.
        num_arrs = len(dataset_arrs)
        datasets_stored = []
        try:
            logging.info('Storing of \'%s\' dataset...', dataset_name)
            dataset_source = columns[6]
            for idx, dataset_arr in enumerate(dataset_arrs):
                if num_arrs > 1:
                    name = dataset_name + '_' + str(idx) # derive unique name for each array
                else:
                    name = dataset_name
                store_dataset(name, dataset_source, col_types, dataset_arr, num_missing_val)
                datasets_stored.append(name)
            logging.info('Completed storing of \'%s\' dataset.', dataset_name)
        except:
            if len(datasets_stored) > 0:
                delete_datasets(datasets_stored) # delete already stored arrays associated with the dataset
            logging.exception('\'%s\' dataset cannot be stored.', dataset_name)
            write_rows.append(','.join(columns))
            continue

        columns[0] = str(1) # update column 0 to True for datasets that are processed and stored successfully
        write_rows.append(','.join(columns))

    # Write updates to formatting information CSV file
    write_formatting_info = '\n'.join(write_rows)
    with open(FORMATTING_INFO_FILE_PATH, 'w', encoding='utf-8') as formatting_info_file:
        formatting_info_file.write(write_formatting_info)

if __name__ == '__main__':
    main()
