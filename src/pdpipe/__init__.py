"""
Easy pandas piplines.

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
    ApplicationContextEnricher,
)

core.__load_stage_attributes_from_module__("pdpipe.basic_stages")

from . import col_generation
from .col_generation import (
    Bin,
    OneHotEncode,
    MapColVals,
    ApplyToRows,
    ApplyByCols,
    TransformByCols,
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
from . import lbl
from .lbl import (
    DropLabelsByValues,
)

core.__load_stage_attributes_from_module__("pdpipe.lbl")

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
    from . import skintegrate
    from . import sklearn_stages
    from .sklearn_stages import (
        Encode,
        Scale,
        TfidfVectorizeTokenLists,
        Decompose,
        EncodeLabel,
    )

    core.__load_stage_attributes_from_module__("pdpipe.sklearn_stages")
except ImportError:
    tb = traceback.format_exc()
    warnings.warn(tb)
    warnings.warn(
        "pdpipe: Scikit-learn or skutil import failed. Scikit-learn"
        "-dependent pipeline stages will not be loaded."
    )


__all__ = [
    "basic_stages",
    "PdPipelineStage",
    "AdHocStage",
    "PdPipeline",
    "make_pdpipeline",
    "ColDrop",
    "ValDrop",
    "ValKeep",
    "ColRename",
    "DropNa",
    "SetIndex",
    "FreqDrop",
    "ColReorder",
    "RowDrop",
    "Schematize",
    "DropDuplicates",
    "ColumnDtypeEnforcer",
    "ConditionValidator",
    "ApplicationContextEnricher",
    "col_generation",
    "Bin",
    "OneHotEncode",
    "MapColVals",
    "ApplyToRows",
    "ApplyByCols",
    "TransformByCols",
    "ColByFrameFunc",
    "AggByCols",
    "Log",
    "text_stages",
    "RegexReplace",
    "DropTokensByLength",
    "DropTokensByList",
    "lbl",
    "DropLabelsByValues",
    "wrappers",
    "FitOnly",
    "fly",
    "drop_rows_where",
    "keep_rows_where",
    "skintegrate",
    "sklearn_stages",
    "Encode",
    "Scale",
    "TfidfVectorizeTokenLists",
    "Decompose",
    "EncodeLabel",
]


try:
    from . import nltk_stages
    from .nltk_stages import (
        TokenizeText,
        UntokenizeText,
        RemoveStopwords,
        SnowballStem,
        DropRareTokens,
    )

    __all__.extend(
        [
            "nltk_stages",
            "TokenizeText",
            "UntokenizeText",
            "RemoveStopwords",
            "SnowballStem",
            "DropRareTokens",
        ]
    )
    core.__load_stage_attributes_from_module__("pdpipe.nltk_stages")
except ImportError:
    tb = traceback.format_exc()
    warnings.warn(tb)
    warnings.warn(
        "pdpipe: nltk import failed. nltk-dependent  pipeline "
        "stages will not be loaded."
    )

from . import run_time_parameters
from .run_time_parameters import dynamic

from .df import DF_HANDLE as df

from . import cq
from . import rq
from . import cond

__all__.extend(
    [
        "run_time_parameters",
        "dynamic",
        "df",
        "cq",
        "rq",
        "cond",
    ]
)


from ._version import __version__

__all__.extend(
    [
        "__version__",
    ]
)

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

from .cfg import (
    LOAD_CORE_AS_MODULE,
)

if LOAD_CORE_AS_MODULE:
    from . import core
del LOAD_CORE_AS_MODULE

# this dictates which modules are skipped on pdoc documentation generation
__pdoc__ = {
    "shared": False,
}
