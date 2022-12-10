"""Main module for meta-learning system CLI.
"""
import logging
import warnings

from rich import print
from rich.console import Console
from rich.table import Table
import typer

from ..datasets import get_datasets, add_dataset, delete_dataset
from ..meta_features import generate_meta_features
from ..meta_labels import generate_meta_labels

app = typer.Typer()
app_dataset = typer.Typer()
app.add_typer(app_dataset, name='dataset')
app_metafeature = typer.Typer()
app.add_typer(app_metafeature, name='metafeature')
app_metalabel = typer.Typer()
app.add_typer(app_metalabel, name='metalabel')

console = Console()

@app_dataset.command('list')
def dataset_list():
    """List all datasets.
    """
    l = get_datasets()
    table = Table('Name', 'Source')
    for t in l:
        table.add_row(t[0], t[1])
    console.print(table)

@app_dataset.command('add')
def dataset_add(
    name: str = typer.Argument(..., help='Dataset name'),
    info: typer.FileText = typer.Argument(..., help='File path of dataset information containing column types'),
    file: typer.FileText = typer.Argument(..., help='File path of dataset'),
    source: str = typer.Argument('user', help='Dataset source')):
    """Add a dataset.
    """
    add_dataset(name, source, info.read(), file.read())
    print('Dataset added successfully.')

@app_dataset.command('delete')
def delete(name: str = typer.Argument(..., help='Dataset name')):
    """Delete a dataset.
    """
    delete_dataset(name)
    print('Dataset deleted successfully.')

@app_metafeature.command('generate')
def metafeature_generate():
    """Generate meta-features.
    """
    generate_meta_features()
    print('Meta-features generated successfully.')

@app_metalabel.command('generate')
def metalabel_generate():
    """Generate meta-labels.
    """
    generate_meta_labels()
    print('Meta-labels generated successfully.')

@app.command()
def end():
    raise typer.Exit()

if __name__ == '__main__':
    # Reset debug log file
    debug_file = open('meta_learning_system/results/debug.log', 'w', encoding='utf-8')
    debug_file.write('')
    debug_file.close()

    # Setup logging
    logging.basicConfig(
        filename='debug.log',
        format='%(levelname)s: %(message)s',
        encoding='utf-8',
        level=logging.INFO)

    warnings.filterwarnings("ignore")

    app()
