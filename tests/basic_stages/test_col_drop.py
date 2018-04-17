"""Testing basic pipeline stages."""

import pandas as pd
import pytest

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


def test_coldrop_missing_col():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    assert 'num1' in df.columns
    stage = ColDrop('num3')
    with pytest.raises(FailedPreconditionError):
        res_df = stage.apply(df)
    stage = ColDrop('num3', errors='ignore')
    res_df = stage.apply(df)
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


def test_coldrop_lambda():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    assert 'num1' in df.columns
    assert 'num2' in df.columns
    stage = ColDrop(lambda col: 'num' in col.name)
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
