"""Testing basic pipeline stages."""

import pandas as pd
import pytest

from pdpipe.core import (
    PdPipelineStage,
    FailedPreconditionError,
    FailedPostconditionError,
)


def _test_df():
    return pd.DataFrame(
        data=[[1, 'a'], [2, 'b']],
        index=[1, 2],
        columns=['num', 'char']
    )


class SomeStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        return df


def test_basic_pipeline_stage():
    """Testing the col binner helper class."""
    test_stage = SomeStage()
    assert not test_stage._is_fittable()
    df = _test_df()
    res_df = test_stage.apply(df, verbose=True)
    assert res_df.equals(df)
    res_df = test_stage(df)
    assert res_df.equals(df)
    res_df = test_stage.apply(df, exraise=True)
    assert res_df.equals(df)
    res_df = test_stage.apply(df, exraise=False)
    assert res_df.equals(df)

    # fit_transform
    res_df = test_stage.fit_transform(df)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, exraise=False)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, exraise=False, verbose=False)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, exraise=False, verbose=True)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, exraise=True)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, exraise=True, verbose=False)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, exraise=True, verbose=True)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, verbose=False)
    assert res_df.equals(df)
    res_df = test_stage.fit_transform(df, verbose=True)
    assert res_df.equals(df)

    expected_repr = f"PdPipelineStage: {PdPipelineStage._DEF_DESCRIPTION}"
    assert str(test_stage) == expected_repr
    assert repr(test_stage) == expected_repr


def test_prec_arg():
    test_stage = SomeStage(prec=lambda df: False)
    df = _test_df()
    with pytest.raises(FailedPreconditionError):
        test_stage(df)


class FailStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prec(self, df):
        return False

    def _transform(self, df, verbose):
        return df


def test_fail_pipeline_stage():
    """Testing the col binner helper class."""
    fail_stage = FailStage()
    df = _test_df()
    with pytest.raises(FailedPreconditionError):
        fail_stage.apply(df, verbose=True)
    with pytest.raises(FailedPreconditionError):
        fail_stage.fit_transform(df, verbose=True)
    res_df = fail_stage.fit_transform(df, exraise=False)
    assert res_df.equals(df)


class SilentDropStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname, **kwargs):
        self.colname = colname
        super().__init__(exraise=False, **kwargs)

    def _prec(self, df):
        return self.colname in df.columns

    def _transform(self, df, verbose):
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


def test_skip_arg():
    silent_fail_stage = SilentDropStage('num')
    df = _test_df()
    assert 'num' in df.columns
    res = silent_fail_stage(df)
    assert 'num' not in res.columns

    silent_fail_stage = SilentDropStage('num', skip=lambda df: True)
    res = silent_fail_stage(df)
    assert 'num' in res.columns


def test_stage_name():
    stage = SilentDropStage('Tigers', name='Name')
    assert stage._name == 'Name'
    with pytest.raises(ValueError) as e:
        stage = SilentDropStage('Tigers', name=12345)
    assert str(e.value) == "'name' must be a str, not int."


class FittableDropByCharStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, char, **kwargs):
        self.char = char
        super().__init__(**kwargs)

    def _prec(self, df):
        return True

    def _fit_transform(self, df, verbose):
        self.columns = [
            x for x in df.columns
            if x.startswith(self.char)
        ]
        keep_cols = [
            x for x in df.columns
            if x not in self.columns
        ]
        self.is_fitted = True
        return df[keep_cols]

    def _transform(self, df, verbose):
        keep_cols = [
            x for x in df.columns
            if x not in self.columns
        ]
        return df[keep_cols]


def _test_df2():
    return pd.DataFrame(
        data=[[1, 'a'], [2, 'b']],
        index=[1, 2],
        columns=['abo', 'coo']
    )


def _test_df3():
    return pd.DataFrame(
        data=[[1, 'a'], [2, 'b']],
        index=[1, 2],
        columns=['abo', 'aoo']
    )


def test_fittable_stage():
    stage = FittableDropByCharStage('a')
    assert stage._is_fittable()

    res1 = stage(_test_df2())
    assert 'abo' not in res1.columns
    assert 'coo' in res1.columns

    res2 = stage(_test_df3())
    assert 'abo' not in res2.columns
    assert 'aoo' in res2.columns

    stage = FittableDropByCharStage('a')
    res3 = stage(_test_df2())
    assert 'abo' not in res3.columns
    assert 'aoo' not in res3.columns


def _no_a_in_cols(df):
    for lbl in df.columns:
        if 'a' in lbl:
            return False
    return True


def test_fittable_stage_with_postcond():
    stage = FittableDropByCharStage('a', post=_no_a_in_cols)
    assert stage._is_fittable()

    res1 = stage(_test_df2())
    assert 'abo' not in res1.columns
    assert 'coo' in res1.columns

    with pytest.raises(FailedPostconditionError):
        stage(_test_df3())

    with pytest.raises(FailedPostconditionError):
        stage.transform(_test_df3())


def test_failing_postcond():
    stage = SomeStage(post=_no_a_in_cols)

    with pytest.raises(FailedPostconditionError):
        stage(_test_df2())

    res = stage(_test_df2(), exraise=False)
    assert 'abo' in res.columns
    assert 'coo' in res.columns

    with pytest.raises(FailedPostconditionError):
        stage.fit_transform(_test_df2(), exraise=True)

    with pytest.raises(FailedPostconditionError):
        stage.fit(_test_df2(), exraise=True)

    with pytest.raises(FailedPostconditionError):
        stage.transform(_test_df2(), exraise=True)
