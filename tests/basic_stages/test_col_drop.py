"""Testing basic pipeline stages."""

import pandas as pd
import pytest

import pdpipe as pdp
from pdpipe import ColDrop
from pdpipe.exceptions import FailedPreconditionError


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


def test_coldrop_one_col():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    assert 'num1' in df.columns
    stage = ColDrop('num1')
    res_df = stage.apply(df)
    assert 'num1' not in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns

    # make sure fit is null operation for unfittable stages
    res_df = stage.fit(df)
    assert 'num1' in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns

    res_df = stage.fit(df, verbose=True)
    assert 'num1' in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns

    # make sure transform and fit_transform are equivalent to apply
    # for unfittable stages
    res_df = stage.transform(df)
    assert 'num1' not in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns

    res_df = stage.transform(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns

    res_df = stage.fit_transform(df)
    assert 'num1' not in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns

    res_df = stage.fit_transform(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns


def test_coldrop_missing_col():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    assert 'num1' in df.columns
    stage = ColDrop('num3')
    with pytest.raises(FailedPreconditionError):
        res_df = stage.apply(df)

    res_df = stage.apply(df, exraise=False)
    assert res_df.equals(df)

    # make sure fit is null operation for unfittable stages
    with pytest.raises(FailedPreconditionError):
        res_df = stage.fit(df)

    with pytest.raises(FailedPreconditionError):
        res_df = stage.fit(df, verbose=True)

    res_df = stage.fit(df, exraise=False)
    assert res_df.equals(df)

    # make sure transform and fit_transform are equivalent to apply
    # for unfittable stages
    with pytest.raises(FailedPreconditionError):
        res_df = stage.transform(df)

    with pytest.raises(FailedPreconditionError):
        res_df = stage.transform(df, verbose=True)

    res_df = stage.transform(df, exraise=False)
    assert res_df.equals(df)

    with pytest.raises(FailedPreconditionError):
        res_df = stage.fit_transform(df)

    with pytest.raises(FailedPreconditionError):
        res_df = stage.fit_transform(df, verbose=True)

    res_df = stage.fit_transform(df, exraise=False)
    assert res_df.equals(df)

    stage = ColDrop('num3', errors='ignore')
    res_df = stage.apply(df)
    assert res_df.equals(df)

    # make sure fit is null operation for unfittable stages
    res_df = stage.fit(df)
    assert res_df.equals(df)

    # make sure transform and fit_transform are equivalent to apply
    # for unfittable stages
    res_df = stage.fit_transform(df)
    assert res_df.equals(df)

    res_df = stage.transform(df)
    assert res_df.equals(df)


def test_coldrop_multi_col():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    assert 'num1' in df.columns
    assert 'num2' in df.columns
    stage = ColDrop(['num1', 'num2'])
    res_df = stage.apply(df)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_coldrop_col_qualifier():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    assert 'num1' in df.columns
    assert 'num2' in df.columns
    stage = ColDrop(pdp.cq.StartWith('num'))
    res_df = stage.apply(df)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def _test_df2():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 2, False]
    )


def test_coldrop_non_str_lbl():
    """Testing the ColDrop pipeline stage."""
    df = _test_df2()
    assert 2 in df.columns
    stage = ColDrop(2)
    res_df = stage.apply(df)
    assert 2 not in res_df.columns
    assert 'num1' in res_df.columns
    assert False in res_df.columns
