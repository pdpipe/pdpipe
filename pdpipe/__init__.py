"""The base package of the pdpipe library."""
# flake8: noqa  # prevents 'imported but unused' erros

# ignore IPython's ShimWarning, if IPython is installed
try:
    import warnings
    from IPython.utils.shimmodule import ShimWarning
    warnings.simplefilter("ignore", ShimWarning)
except ImportError:
    pass



from . import core
from . import basic_stages
import sys
core.__load_stage_attributes__(sys.modules[__name__])
del core
del basic_stages
del sys

from .core import *
from .basic_stages import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
