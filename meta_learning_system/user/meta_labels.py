"""Meta-label user methods.
"""
from ..system.meta_labels.main import main_user

def generate_meta_labels():
    """Generate meta-labels for new dataset(s).

    Raises:
        Exception: If generation of meta-labels for any new dataset has failed.
    """
    try:
        main_user()
    except:
        raise Exception(
            'Meta-labels for new dataset(s) cannot be generated. Please check the debug log for details.'
        ) from None
