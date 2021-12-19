"""Handles for dynamic dataframe-method-wrapping pipeline stages.

All `pandas.DataFrame` methods can be used as stages using this module.

For example `pdp.df.dropna(axis=1)` will return a `pdpipe.PdPipelineStage`
object that will call the `dropna` method of input DataFrames with the `axis=1`
keyword argument provided, and return the resulting dataframe object
(practically dropping any column with a missing value from the input
dataframe).

These stage combine naturally into `pdpipe` pipelines:

    >>> import pdpipe as pdp;
    >>> pipeline = pdp.PdPipeline([
    ...     pdp.df.set_index(keys='datetime'),
    ...     pdp.ColDrop('age'),
    ... ])

There are a couple of caveats:

* `pdpipe` pipeline stages never alter input dataframes, so the `inplace`
keyword argument is always ignored, even if provided.
* All method parameters are fixed on pipeline stage creation time, and must be
explicitly provided as keyword arguments, and not as positional ones.
"""


from typing import Dict

from pandas import DataFrame

from .core import PdPipelineStage


# this_module = __import__(__name__)


class _DataFrameMethodTransformer(PdPipelineStage):

    def __init__(self, method_name: str, kwargs: Dict[str, object]) -> None:
        self._method_name = method_name
        self._kwargs = kwargs.copy()
        # we must always pop 'inplace', if it's there
        found = self._kwargs.pop('inplace', None)
        if found is not None:
            self._kwargs['inplace'] = False
        exmsg = (
            "Pipeline stage failed while applying method {} with kwargs {}"
        ).format(method_name, self._kwargs)
        desc = "Apply dataframe method {} with kwargs {}".format(
            method_name, self._kwargs)
        super_kwargs = {
            'exmsg': exmsg,
            'desc': desc,
            'name': method_name,
        }
        super().__init__(**super_kwargs)

    def _prec(self, df: DataFrame) -> bool:  # pragma: no cover
        return True

    def _transform(self, df: DataFrame, verbose: bool) -> DataFrame:
        method = getattr(df, self._method_name)
        return method(**self._kwargs)


class _DfMethodTransformerHandle(object):

    def __init__(self, method_name: str, doc: str) -> None:
        self._method_name = method_name
        self.__doc__ = doc

    def __call__(self, **kwargs: Dict[str, object]) -> PdPipelineStage:
        return _DataFrameMethodTransformer(
            method_name=self._method_name,
            kwargs=kwargs,
        )


__RETURNS = 'Returns'
__DATAFRAME = 'DataFrame'


def _is_dataframe_transform(attr_name: str, attr: object) -> bool:
    if attr_name.startswith('_') or not callable(attr):
        return False
    try:
        doc_lines = attr.__doc__.split('\n')
        returns_line_index = None
        for i in range(len(doc_lines)):
            if __RETURNS in doc_lines[i]:
                returns_line_index = i + 2
                break
        if returns_line_index:
            return_type_line = doc_lines[returns_line_index]
            if __DATAFRAME in return_type_line:
                return True
        return False
    except (AttributeError, IndexError):  # pragma: no cover
        return False


for attr_name in dir(DataFrame):
    attr = getattr(DataFrame, attr_name)
    if _is_dataframe_transform(attr_name, attr):
        # print(f"Adding {attr_name} of {attr}")
        handle = _DfMethodTransformerHandle(
            method_name=attr_name,
            doc=attr.__doc__,
        )
        globals()[attr_name] = handle
        # setattr(this_module, attr_name, handle)

# print(this_module)
# del this_module
del attr
del handle
del attr_name
del Dict
del DataFrame
del PdPipelineStage
