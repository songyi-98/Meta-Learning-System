"""Main module for the meta-learning system.
"""
import logging

if __name__ == '__main__':
    # Reset debug log file
    debug_file = open('meta_learning_system/debug.log', 'w', encoding='utf-8')
    debug_file.write('')
    debug_file.close()

    # Setup logging
    logging.basicConfig(
        filename='debug.log',
        format='%(levelname)s: %(message)s',
        encoding='utf-8',
        level=logging.INFO)

    # Datasets pipeline
    logging.info('START - DATASETS PIPELINE')
    from .datasets import main as datasets
    datasets.main()
    logging.info('END - DATASETS PIPELINE')

    # Meta-features pipeline
    logging.info('START - META-FEATURES PIPELINE')
    from .meta_features import main as meta_features
    meta_features.main()
    logging.info('END - META-FEATURES PIPELINE')

    # Meta-labels pipeline
    logging.info('START - META-LABELS PIPELINE')
    from .meta_labels import main as meta_labels
    meta_labels.main()
    logging.info('END - META-LABELS PIPELINE')
