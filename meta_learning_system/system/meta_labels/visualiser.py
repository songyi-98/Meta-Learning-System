"""Visualise meta-labels.
"""
import matplotlib.pyplot as plt

from .models.main import MODELS, SCORING

IMAGES_PATH = 'meta_learning_system/results/'

def visualise_meta_labels(meta_labels):
    """Visualise meta-labels for all datasets.

    Args:
        meta_labels (NDArray): Array of meta-labels for all datasets with shape (num_datasets, num_metrics, num_models).
    """
    _, num_metrics, num_models = meta_labels.shape
    for idx in range(num_metrics):
        metric_name = SCORING[idx]
        # Array of all models' scores for a particular evaluation metric for all datasets
        # with shape (num_models, num_datasets)
        models_metric_scores = meta_labels[:, idx, :].T

        # TODO: Configure to allow combinations of models for comparison
        assert num_models == 2
        fig, ax = plt.subplots()
        ax.scatter(models_metric_scores[0], models_metric_scores[1])
        ax.set_xlabel(MODELS[0])
        ax.set_xbound(0.0, 1.0)
        ax.set_ylabel(MODELS[1])
        ax.set_ybound(0.0, 1.0)
        ax.set_title('{} VS {}'.format(MODELS[0], MODELS[1]))
        ax.grid(True)
        fig.tight_layout()
        plt.savefig(IMAGES_PATH + f'{metric_name}_{MODELS[0]}_{MODELS[1]}.png')
