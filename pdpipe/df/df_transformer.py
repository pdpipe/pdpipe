"""Handles for all pandas.DataFrame dataframe-outputing transformations."""

from typing import List, Dict

from pandas import DataFrame

from ..core import PdPipelineStage


# === DataFrame methods  ===

class _DataFrameMethodTransformer(PdPipelineStage):

    def __init__(
        self,
        method_name: str,
        args: List[object],
        kwargs: Dict[str, object],
    ) -> None:
        self._method_name = method_name
        self._args = args
        self._kwargs = kwargs
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
        return method(*self._args, **self._kwargs)


class _DfMethodTransformerHandle():

    def __init__(self, method_name: str, doc: str) -> None:
        self._method_name = method_name
        self.__doc__ = doc

    def __call__(self, *args, **kwargs: Dict[str, object]) -> PdPipelineStage:
        return _DataFrameMethodTransformer(
            method_name=self._method_name,
            args=args,
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
        for i, line in enumerate(doc_lines):
            if __RETURNS in line:
                returns_line_index = i + 2
                break
        if returns_line_index:
            return_type_line = doc_lines[returns_line_index]
            if __DATAFRAME in return_type_line:
                return True
        return False
    except (AttributeError, IndexError):  # pragma: no cover
        return False


# === END-OF DataFrame methods  ===
