"""Testing basic pipline stages."""

import pandas as pd
import pytest

from pdpipe.core import (
    PipelineStage,
    FailedPreconditionError
)


def _test_df():
    return pd.DataFrame(
        data=[[1, 'a'], [2, 'b']],
        index=[1, 2],
        columns=['num', 'char']
    )


class SomeStage(PipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prec(self, df):
        return True

    def _op(self, df, verbose):
        return df


def test_basic_pipeline_stage():
    """Testing the col binner helper class."""
    test_stage = SomeStage()
    df = _test_df()
    res_df = test_stage.apply(df, verbose=True)
    assert res_df.equals(df)
    res_df = test_stage(df)
    assert res_df.equals(df)
    res_df = test_stage.apply(df, exraise=True)
    assert res_df.equals(df)
    res_df = test_stage.apply(df, exraise=False)
    assert res_df.equals(df)
    assert str(test_stage) == PipelineStage._DEF_DESCRIPTION
    assert repr(test_stage) == PipelineStage._DEF_DESCRIPTION


class FailStage(PipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prec(self, df):
        return False

    def _op(self, df, verbose):
        return df


def test_fail_pipeline_stage():
    """Testing the col binner helper class."""
    fail_stage = FailStage()
    df = _test_df()
    with pytest.raises(FailedPreconditionError):
        fail_stage.apply(df, verbose=True)


class SilentDropStage(PipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname):
        self.colname = colname
        super().__init__(exraise=False)

    def _prec(self, df):
        return self.colname in df.columns

    def _op(self, df, verbose):
        return df.drop([self.colname], axis=1)


def test_silent_fail_pipeline_stage():
    """Testing the col binner helper class."""
    silent_fail_stage = SilentDropStage('Tigers')
    df = _test_df()
    res_df = silent_fail_stage.apply(df, verbose=True)
    assert res_df.equals(df)


def test_pipeline_stage_addition_to_int():
    """Testing that """
    silent_fail_stage = SilentDropStage('Tigers')
    with pytest.raises(TypeError):
        silent_fail_stage + 2
