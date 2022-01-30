"""
The `pdpipe` Python package provides a concise interface for building `pandas`
pipelines that have pre-conditions, are verbose, support the fit-transform
design of scikit-learn transformers and are highly serializable. `pdpipe`
pipelines have a simple interface, informative prints and errors on pipeline
application, support pipeline arithmetics and enable easier handling of
mixed-type data.

.. include:: ./documentation.md
"""
# pylint: disable=C0413
# flake8: noqa

import warnings
import traceback


from . import core
from .core import PdPipelineStage, AdHocStage, PdPipeline, make_pdpipeline

core.__load_stage_attributes_from_module__("pdpipe.core")

from . import basic_stages
from .basic_stages import (
    ColDrop,
    ValDrop,
    ValKeep,
    ColRename,
    DropNa,
    SetIndex,
    FreqDrop,
    ColReorder,
    RowDrop,
    Schematize,
    DropDuplicates,
    ColumnDtypeEnforcer,
    ConditionValidator,
)

core.__load_stage_attributes_from_module__("pdpipe.basic_stages")

from . import col_generation
from .col_generation import (
    Bin,
    OneHotEncode,
    MapColVals,
    ApplyToRows,
    ApplyByCols,
    ColByFrameFunc,
    AggByCols,
    Log,
)

core.__load_stage_attributes_from_module__("pdpipe.col_generation")

from . import text_stages
from .text_stages import (
    RegexReplace,
    DropTokensByLength,
    DropTokensByList,
)

core.__load_stage_attributes_from_module__("pdpipe.text_stages")

from . import wrappers
from .wrappers import (
    FitOnly,
)

core.__load_stage_attributes_from_module__("pdpipe.wrappers")

from .fly import (
    drop_rows_where,
    keep_rows_where,
)

try:
    from . import sklearn_stages
    from .sklearn_stages import (
        Encode,
        Scale,
        TfidfVectorizeTokenLists,
    )

    core.__load_stage_attributes_from_module__("pdpipe.sklearn_stages")
except ImportError:
    tb = traceback.format_exc()
    warnings.warn(tb)
    warnings.warn(
        "pdpipe: Scikit-learn or skutil import failed. Scikit-learn"
        "-dependent pipeline stages will not be loaded."
    )

try:
    from . import nltk_stages
    from .nltk_stages import (
        TokenizeText,
        UntokenizeText,
        RemoveStopwords,
        SnowballStem,
        DropRareTokens,
    )

    core.__load_stage_attributes_from_module__("pdpipe.nltk_stages")
except ImportError:
    tb = traceback.format_exc()
    warnings.warn(tb)
    warnings.warn(
        "pdpipe: nltk import failed. nltk-dependent  pipeline "
        "stages will not be loaded."
    )


from .df import DF_HANDLE as df

from . import cq
from . import rq
from . import cond
from . import skintegrate


from ._version import get_versions

__version__ = get_versions()["version"]

for name in [
    "warnings",
    "traceback",
    "_custom_formatwarning",
    "core",
    "basic_stages",
    "sklearn_stages",
    "col_generation",
    "shared",
    "util",
    "_version",
    "get_versions",
    "THE_DF_HANDLE",
]:
    try:
        globals().pop(name)
    except KeyError:
        pass
try:
    del name  # pylint: disable=W0631
except NameError:
    pass

# this dictates which modules are skipped on pdoc documentation generation
__pdoc__ = {
    'shared': False,
}

from . import _version
__version__ = _version.get_versions()['version']
