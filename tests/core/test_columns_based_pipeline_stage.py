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

    def _transformation(self, df, verbose, fit):
        return df.drop(
            self._get_columns(df, fit=fit), axis=1, errors=self._errors)


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
    stage = Drop(lambda df: [
        lbl for lbl in df.columns if _safe_start_with(lbl)
    ])
    res = stage(df1)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns

    stage = Drop(5)
    with pytest.raises(FailedPreconditionError):
        stage(df1)

    with pytest.raises(ValueError):
        Drop(None)


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

    def _transformation(self, df, verbose, fit):
        return df.drop(
            self._get_columns(df, fit=fit), axis=1, errors=self._errors)


def test_columns_based_stage2():
    df1 = _df1()
    stage = Drop2('num')
    res = stage(df1)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns


class Double(ColumnsBasedPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, columns, errors=None, **kwargs):
        self._errors = errors
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop columns {}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'all'
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        inter_df = df
        for col in self._get_columns(df, fit=fit):
            inter_df[col] = inter_df[col] * 2
        return inter_df


def _df_d():
    return pd.DataFrame(
        data=[[1, 2, 3]],
        index=[1],
        columns=['num', 'char', 4]
    )


def test_columns_based_stage_none():
    df = _df_d()
    stage = Double(None)
    res = stage(df)
    assert res.iloc[0, 0] == 2
    assert res.iloc[0, 1] == 4
    assert res.iloc[0, 2] == 6

    stage = Double('num')
    df = _df_d()
    res = stage(df)
    assert res.iloc[0, 0] == 2
    assert res.iloc[0, 1] == 2
    assert res.iloc[0, 2] == 3


class BadNoneColumnsStrArg(ColumnsBasedPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, columns, errors=None, **kwargs):
        self._errors = errors
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop columns {}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'badval'
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        inter_df = df
        for col in self._get_columns(df, fit=fit):
            inter_df[col] = inter_df[col] * 2
        return inter_df


def test_bad_none_columns_str_arg():
    with pytest.raises(ValueError):
        BadNoneColumnsStrArg(['collbl'])


class BadNoneColumnsArg(ColumnsBasedPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, columns, errors=None, **kwargs):
        self._errors = errors
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop columns {}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = FailedPreconditionError()
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        inter_df = df
        for col in self._get_columns(df, fit=fit):
            inter_df[col] = inter_df[col] * 2
        return inter_df


def test_bad_none_columns_arg():
    with pytest.raises(ValueError):
        BadNoneColumnsArg(['collbl'])


class NumDefaultDrop(ColumnsBasedPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, columns, errors=None, **kwargs):
        self._errors = errors
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop columns {}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = ['num']
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        keep_cols = [
            x for x in df.columns
            if x not in self._get_columns(df, fit=fit)
        ]
        return df[keep_cols]


def test_columns_based_stage_none_columns_is_list():
    df = _df_d()
    stage = NumDefaultDrop('char')
    res = stage(df)
    assert 'num' in res.columns
    assert 'char' not in res.columns
    assert 4 in res.columns

    stage = NumDefaultDrop(None)
    res = stage(df)
    assert 'num' not in res.columns
    assert 'char' in res.columns
    assert 4 in res.columns
