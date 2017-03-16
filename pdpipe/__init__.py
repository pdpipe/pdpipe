"""Easy pipelines for pandas."""
# flake8: noqa  # prevents 'imported but unused' erros
# pylint: disable=C0413

import sys
from . import core
from . import basic_stages
core.__load_stage_attributes__()

from .core import (
    PipelineStage,
    AdHocStage,
    Pipeline
)
from .basic_stages import (
    ColDrop,
    ValDrop,
    ValKeep,
    ColRename,
    Bin,
    Binarize,
    MapColVals,
    Encode,
    ColByFunc
)

from ._version import get_versions
__version__ = get_versions()['version']

for name in ['_version', 'get_versions', 'core', 'basic_stages', 'sys']:
    try:
        globals().pop(name)
    except KeyError:
        pass
try:
    del name  # pylint: disable=W0631
except NameError:
    pass
