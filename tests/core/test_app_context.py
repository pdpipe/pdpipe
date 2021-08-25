"""Testing application context.."""

from random import randint
from builtins import ValueError

import pandas as pd
import pytest

from pdpipe.core import (
    PdPipelineStage,
    PdPipeline
)
from pdpipe import make_pdpipeline, ColByFrameFunc, ColDrop
from pdpipe.exceptions import PipelineApplicationError
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
        self._fit_context['a'] = self.randi
        return df.drop([self.colname], axis=1)


class UseContextStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname, **kwargs):
        self.colname = colname
        super().__init__(exraise=False, **kwargs)

    def _prec(self, df):
        return self.colname in df.columns

    def _transform(self, df, verbose):
        val = self._fit_context['a']
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

    val2 = pipeline._fit_context['a']
    assert val2 == val
    # check locking works
    pipeline._fit_context['a'] = 0
    val2 = pipeline._fit_context['a']
    assert val2 == val
    del pipeline._fit_context['a']
    val2 = pipeline._fit_context['a']
    assert val2 == val
    pipeline._fit_context.pop('a', 0)
    val2 = pipeline._fit_context['a']
    assert val2 == val
    pipeline._fit_context.clear()
    val2 = pipeline._fit_context['a']
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
