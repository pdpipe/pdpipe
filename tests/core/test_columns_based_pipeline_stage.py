"""Testing columns-based pipeline stages."""

import pandas as pd
import pytest

from pdpipe.core import (
    ColumnsBasedPipelineStage,
    FailedPreconditionError
)


class Drop(ColumnsBasedPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, columns, errors=None, **kwargs):
        self._errors = errors
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop columns {}',
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _fit_transform(self, df, verbose):
        return df.drop(
            self._get_columns(df, fit=True), axis=1, errors=self._errors)

    def _transform(self, df, verbose):
        return df.drop(
            self._get_columns(df, fit=False), axis=1, errors=self._errors)


def _df1():
    return pd.DataFrame(
        data=[[1, 'a', 8], [2, 'b', 9]],
        index=[1, 2],
        columns=['num', 'char', 4]
    )


def test_columns_based_stage():
    df1 = _df1()
    stage = Drop('num')
    res = stage(df1)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns

    stage = Drop(4)
    res = stage(df1)
    assert 'num' in res.columns
    assert 'char' in res.columns
    assert 4 not in res.columns

    stage = Drop([4])
    res = stage(df1)
    assert 'num' in res.columns
    assert 'char' in res.columns
    assert 4 not in res.columns

    stage = Drop(['num'])
    res = stage(df1)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns

    def _safe_start_with(string):
        try:
            return string.startswith('n')
        except AttributeError:
            return False
    stage = Drop(lambda df: [l for l in df.columns if _safe_start_with(l)])
    res = stage(df1)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns

    stage = Drop(5)
    with pytest.raises(FailedPreconditionError):
        stage(df1)


class Drop2(ColumnsBasedPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, columns, errors=None, **kwargs):
        self._errors = errors
        super_kwargs = {
            'columns': columns,
            'exmsg': 'Error!',
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _fit_transform(self, df, verbose):
        return df.drop(
            self._get_columns(df, fit=True), axis=1, errors=self._errors)

    def _transform(self, df, verbose):
        return df.drop(
            self._get_columns(df, fit=False), axis=1, errors=self._errors)


def test_columns_based_stage2():
    df1 = _df1()
    stage = Drop2('num')
    res = stage(df1)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns
