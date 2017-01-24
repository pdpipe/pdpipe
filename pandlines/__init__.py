"""The base package of the datapy library."""
# flake8: noqa  # prevents 'imported but unused' erros

# ignore IPython's ShimWarning, if IPython is installed
try:
    import warnings
    from IPython.utils.shimmodule import ShimWarning
    warnings.simplefilter("ignore", ShimWarning)
except ImportError:
    pass

from .core import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
