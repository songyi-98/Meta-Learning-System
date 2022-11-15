"""Meta-feature user methods.
"""
from ..system.meta_features.main import main_user

def generate_meta_features():
    """Generate meta-features for new dataset(s).

    Raises:
        Exception: If generation of meta-features for any new dataset has failed.
    """
    try:
        main_user()
    except:
        raise Exception(
            'Meta-features for new dataset(s) cannot be generated. Please check the debug log for details.'
        ) from None
