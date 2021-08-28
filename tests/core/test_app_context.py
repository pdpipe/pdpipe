"""Testing application context.."""

from random import randint

import pandas as pd
import pytest

from pdpipe.core import (
    PdPipelineStage,
    AdHocStage,
    PdPipeline
)
from pdpipe.util import out_of_place_col_insert
from pdpipe.core import PdpApplicationContext


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


class PutContextStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname, **kwargs):
        self.colname = colname
        self.randi = randint(1, 832)
        super().__init__(exraise=False, **kwargs)

    def _prec(self, df):
        return self.colname in df.columns

    def _transform(self, df, verbose):
        self.fit_context['a'] = self.randi
        return df.drop([self.colname], axis=1)


class UseContextStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname, **kwargs):
        self.colname = colname
        super().__init__(exraise=False, **kwargs)

    def _prec(self, df):
        return self.colname in df.columns

    def _transform(self, df, verbose):
        val = self.fit_context['a']
        source_col = df[self.colname]
        loc = df.columns.get_loc(self.colname) + 1
        series = source_col.apply(lambda x: x + val)
        inter_df = out_of_place_col_insert(
            df=df,
            series=series,
            loc=loc,
            column_name=self.colname + "+val",
        )
        return inter_df


def test_application_context():
    """Testing something."""
    put_context = PutContextStage('char')
    val = put_context.randi
    use_context = UseContextStage('num1')
    pipeline = PdPipeline([put_context, use_context])
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' in res_df.columns
    assert 'num1+val' in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' not in res_df.columns
    num1 = df['num1']
    resnum = res_df['num1+val']
    assert resnum.iloc[0] == val + num1.iloc[0]
    assert resnum.iloc[1] == val + num1.iloc[1]

    val2 = pipeline.fit_context['a']
    assert val2 == val
    # check locking works
    pipeline.fit_context['a'] = 0
    val2 = pipeline.fit_context['a']
    assert val2 == val
    del pipeline.fit_context['a']
    val2 = pipeline.fit_context['a']
    assert val2 == val
    pipeline.fit_context.pop('a', 0)
    val2 = pipeline.fit_context['a']
    assert val2 == val
    pipeline.fit_context.clear()
    val2 = pipeline.fit_context['a']
    assert val2 == val


def test_application_context_unit():
    context = PdpApplicationContext()
    context['a'] = 1
    assert context['a'] == 1
    assert context.pop('a', 0) == 1
    with pytest.raises(KeyError):
        context['a']
    context['a'] = 1
    del context['a']
    with pytest.raises(KeyError):
        context['a']
    context['a'] = 1
    context.clear()
    with pytest.raises(KeyError):
        context['a']
    context.update({'b': 2})
    assert context['b'] == 2
    outer_context = PdpApplicationContext(context)
    assert outer_context.fit_context() == context

    # now check locking
    context['a'] = 1
    context.lock()
    assert context.pop('a', 0) == 1
    assert context['a'] == 1
    del context['a']
    assert context['a'] == 1
    context.clear()
    assert context['a'] == 1
    context.update({'c': 3})
    with pytest.raises(KeyError):
        context['c']


CONTEXT_VAR = 'a'
CONTEXT_NUM = 5
SRC_LBL = 'num1'
TRGT_LBL = 'res'


def put_context(df, fit_context, verbose):
    if verbose:
        print('Verbose in put_context test!')
    fit_context[CONTEXT_VAR] = CONTEXT_NUM
    return df


def use_context(df, fit_context):
    val = fit_context[CONTEXT_VAR]
    source_col = df[SRC_LBL]
    loc = df.columns.get_loc(SRC_LBL) + 1
    series = source_col.apply(lambda x: x + val)
    inter_df = out_of_place_col_insert(
        df=df,
        series=series,
        loc=loc,
        column_name=TRGT_LBL,
    )
    return inter_df


def test_context_with_adhoc_stage():
    pipeline = PdPipeline([
        AdHocStage(transform=put_context),
        AdHocStage(transform=use_context),
    ])
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' in res_df.columns
    assert TRGT_LBL in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns
    num1 = df[SRC_LBL]
    resnum = res_df[TRGT_LBL]
    assert resnum.iloc[0] == CONTEXT_NUM + num1.iloc[0]
    assert resnum.iloc[1] == CONTEXT_NUM + num1.iloc[1]
