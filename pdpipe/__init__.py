"""Easy pipelines for pandas."""
# pylint: disable=C0413

import warnings


from . import core
from .core import (
    PipelineStage,
    AdHocStage,
    Pipeline
)
core.__load_stage_attributes_from_module__('pdpipe.core')

from . import basic_stages
from .basic_stages import (
    ColDrop,
    ValDrop,
    ValKeep,
    ColRename,
    DropNa,
    FreqDrop,
)
core.__load_stage_attributes_from_module__('pdpipe.basic_stages')

from . import col_generation
from .col_generation import (
    Bin,
    Binarize,
    MapColVals,
    ApplyToRows,
    ApplyByCols
)
core.__load_stage_attributes_from_module__('pdpipe.col_generation')

try:
    from . import sklearn_stages
    from .sklearn_stages import (
        Encode
    )
    core.__load_stage_attributes_from_module__('pdpipe.sklearn_stages')
except ImportError:
    warnings.warn("pdpipe: Scikit-learn import failed. Scikit-learn-dependent "
                  " pipeline stages will not be loaded.")

from ._version import get_versions
__version__ = get_versions()['version']

for name in [
        'warnings', '_custom_formatwarning', 'core', 'basic_stages',
        'sklearn_stages', 'col_generation', 'shared', 'util', '_version',
        'get_versions']:
    try:
        globals().pop(name)
    except KeyError:
        pass
try:
    del name  # pylint: disable=W0631
except NameError:
    pass
