"""Testing basic pipeline stages."""

import pandas as pd
import pytest

from pdpipe.core import (
    PipelineStage,
    Pipeline
)


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


class SilentDropStage(PipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname):
        self.colname = colname
        super().__init__(exraise=False)

    def _prec(self, df):
        return self.colname in df.columns

    def _op(self, df, verbose):
        return df.drop([self.colname], axis=1)


def test_two_stage_pipeline_stage():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = Pipeline([drop_num1, drop_num2])
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns
    str(pipeline)


def test_pipeline_stage_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = drop_num1 + drop_num2
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_to_pipeline_stage_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = Pipeline([drop_num1])
    assert len(pipeline) == 1
    pipeline = pipeline + drop_num2
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_stage_to_pipeline_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = Pipeline([drop_num1])
    assert len(pipeline) == 1
    pipeline = drop_num2 + pipeline
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_to_pipeline_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline1 = Pipeline([drop_num1])
    pipeline2 = Pipeline([drop_num2])
    pipeline = pipeline1 + pipeline2
    assert len(pipeline) == 2
    assert pipeline[0] == drop_num1
    assert pipeline[1] == drop_num2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_to_int_addition():
    """Testing something."""
    pipeline = Pipeline([SilentDropStage('num1')])
    with pytest.raises(TypeError):
        res = pipeline + 43
        assert not isinstance(res, Pipeline)


def test_pipeline_index():
    """Testing something."""
    df = _test_df()
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    drop_char = SilentDropStage('char')
    pipeline = Pipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    assert pipeline[0] == drop_num1
    assert 'num1' not in pipeline[0](df).columns
    assert pipeline[1] == drop_num2
    assert 'num2' not in pipeline[1](df).columns
    assert pipeline[2] == drop_char
    assert 'char' not in pipeline[2](df).columns


def test_pipeline_slice():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    drop_char = SilentDropStage('char')
    pipeline = Pipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    pipeline = pipeline[0:2]
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns
