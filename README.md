# Meta-Learning System

## Objectives

1. Collect datasets from available repositories (e.g. UCL) to form a dataset collection.
2. Generate meta-data (meta-features and meta-labels) for each dataset in the dataset collection.
3. Visualise accuracy space between kNN and decision tree.

## Run system

`python -m meta_learning_system.system.main`

## Run user CLI

**NOTE**: Prefix all commands with `python -m meta_learning_system.user.cli.main`. (A Python package needs to be built to remove this requirement.)

**TIP**: `--help` option is available for any commands. (Try `python -m meta_learning_system.user.cli.main --help`.)

### Datasets

1. `dataset list` - list all datasets
2. `dataset add name info file source` - add a dataset
3. `dataset delete name` - delete a dataset

### Meta-data

1. `metafeature generate` - generate meta-features
2. `metalabel generate` - generate meta-labels

### Miscellaneous

1. `exit` - exit CLI

## Libraries

**NOTE**: The list of libraries used in this project is not exhaustive. Only important ones are listed.

1. numpy
2. scipy
3. sklearn
4. skopt
5. typer
