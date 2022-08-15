"""Pre-processing methods for a dataset.

Pre-processing steps include:
1. Standardise data formatting.
2. Perform integer encoding on categorical attributes.
3. Fill missing attribute values.
4. Normalise numerical attributes.
5. Delete instances with missing class labels.
"""
import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder

CURR_DIR = os.path.dirname(__file__)
RAW_DATASETS_FOLDER_PATH = os.path.join(CURR_DIR, 'datasets_raw\\')
COMBINED_FILE_PATH = RAW_DATASETS_FOLDER_PATH + 'combined_file.data'

COL_TYPE_DELETE = 'del'
COL_TYPE_CATEGORICAL_ATTR = 'cat'
COL_TYPE_NUMERICAL_ATTR = 'num'
COL_TYPE_TARGET_VAR = 'target'

MISSING_VALUE = '?'

def combine_files(file_names):
    """Combine multiple files associated with a dataset into a combined file.

    Args:
        file_names (list[str]): List of file name(s) for a dataset.
    """
    with open(COMBINED_FILE_PATH, 'w', encoding='utf-8') as combined_file:
        for file_name in file_names:
            with open(RAW_DATASETS_FOLDER_PATH + file_name, 'r', encoding='utf-8') as file:
                dataset = file.read().rstrip()
                combined_file.write(dataset)
                combined_file.write('\n')

def read_dataset(dataset_file_names):
    """Read dataset as a string.

    Args:
        dataset_file_names (list[str]): List of dataset file names.

    Returns:
        str: Data in dataset.
    """
    if len(dataset_file_names) > 1:
        combine_files(dataset_file_names)
        dataset_file_path = COMBINED_FILE_PATH
    else:
        dataset_file_path = RAW_DATASETS_FOLDER_PATH + dataset_file_names[0]

    with open(dataset_file_path, 'r', encoding='utf-8') as dataset_file:
        dataset = dataset_file.read().strip()
    return dataset

def pre_process_dataset(name, dataset, col_types, is_comma_separator=True):
    """Pre-process a dataset.

    Args:
        name (str): Dataset name.
        dataset (str): Data in dataset.
        col_types (list[str]): List of column types.
        is_comma_separator (bool, optional): True if commas are used to separate values. Defaults to True.

    Raises:
        Exception: If a dataset is invalid.

    Returns:
        tuple[NDArray, list[NDArray], int]: Tuple of
            array of column types with shape (num_attr + 1,),
            list of dataset arrays each of float type and with shape (num_inst, num_attr + 1) and
            number of missing attribute values in a dataset.
    """
    # TODO: Skip dataset with missing values for now
    if '?' in dataset:
        return [], np.array([]), 0

    # Check the number of instances
    rows = dataset.split('\n')
    if len(rows) < 60:
        raise Exception(f'\'{name}\' dataset has less than 60 instances.')

    # Format each row of the dataset and split values in each row based on the column type
    num_columns = len(col_types)
    attrs_cat_transposed = []
    attrs_num_transposed = []
    targets_transposed = []
    num_missing_val = 0
    for row in rows:
        if is_comma_separator:
            values = row.split(',')
        else:
            values = row.split()

        # Check the number of columns
        if len(values) != num_columns:
            raise Exception(f'\'{name}\' dataset has an inconsistent number of columns.')

        attrs_cat_transposed_row = []
        attrs_num_transposed_row = []
        targets_transposed_row = []
        len_count = 0
        for col_idx, value in enumerate(values):
            col_type = col_types[col_idx]
            value = value.strip()
            if value == MISSING_VALUE:
                value = np.nan
                if col_type in [COL_TYPE_CATEGORICAL_ATTR, COL_TYPE_NUMERICAL_ATTR]:
                    num_missing_val += 1

            if col_type == COL_TYPE_DELETE:
                continue # skip column value to be deleted
            if col_type == COL_TYPE_CATEGORICAL_ATTR:
                attrs_cat_transposed_row.append(value)
            elif col_type == COL_TYPE_NUMERICAL_ATTR:
                attrs_num_transposed_row.append(value)
            elif col_type == COL_TYPE_TARGET_VAR:
                targets_transposed_row.append(value)

            len_count += 1

        attrs_cat_transposed.append(attrs_cat_transposed_row)
        attrs_num_transposed.append(attrs_num_transposed_row)
        targets_transposed.append(targets_transposed_row)

    attrs_cat = np.array(attrs_cat_transposed).T # each row is values for ONE categorical attribute
    attrs_num = np.array(attrs_num_transposed).T # each row is values for ONE numerical attribute
    targets = np.array(targets_transposed).T # each row is values for ONE target variable
    num_attrs_cat = len(attrs_cat)
    num_attrs_num = len(attrs_num)

    # Perform integer encoding on each categorical attribute
    if num_attrs_cat > 0:
        attrs_cat = OrdinalEncoder().fit_transform(attrs_cat.T).T

    # Cast numerical attributes array from string to float type
    try:
        attrs_num = attrs_num.astype(np.float64)
    except ValueError as exc:
        raise Exception(
            f'\'{name}\' dataset has numerical attribute value(s) that cannot be converted to floats.'
        ) from exc

    # TODO: Impute missing attribute values

    # Normalise numerical attributes
    if num_attrs_num > 0:
        attrs_num = MinMaxScaler().fit_transform(attrs_num.T).T

    # Duplicate dataset x times if the dataset has x target variables
    dataset_arrs = []
    num_targets = len(targets)
    for target in targets:
        # Remove instances with missing class label
        missing_idxs = np.argwhere(pd.isnull(target)).flatten()
        if num_attrs_cat > 0:
            attrs_cat_new = np.delete(attrs_cat, missing_idxs, axis=1)
        if num_attrs_num > 0:
            attrs_num_new = np.delete(attrs_num, missing_idxs, axis=1)
        target_new = np.delete(target, missing_idxs)

        # Perform integer encoding on class labels
        target_new = LabelEncoder().fit_transform(target_new)
        _, counts = np.unique(target_new, return_counts=True)
        if np.count_nonzero(counts < 2) > 0:
            if num_targets == 1:
                logging.warning('\'%s\' dataset has a class with less than two instances.', name)
            else:
                logging.warning(
                    'One of %i target variables in \'%s\' dataset has a class with less than two instances.',
                    num_targets,
                    name
                )
            continue

        # Build dataset array
        target_new = np.array([target_new])
        if num_attrs_cat == 0:
            dataset_arr = np.concatenate((attrs_num_new, target_new), axis=0).T
        elif num_attrs_num == 0:
            dataset_arr = np.concatenate((attrs_cat_new, target_new), axis=0).T
        else:
            dataset_arr = np.concatenate((attrs_cat_new, attrs_num_new, target_new), axis=0).T
        dataset_arr = dataset_arr.astype(np.float32)
        dataset_arrs.append(dataset_arr)

    # Build column types array
    col_types = np.append(
        np.repeat(COL_TYPE_CATEGORICAL_ATTR, num_attrs_cat),
        np.append(np.repeat(COL_TYPE_NUMERICAL_ATTR, num_attrs_num), COL_TYPE_TARGET_VAR))

    return col_types, dataset_arrs, num_missing_val
