"""Testing basic pipeline stages."""

import datetime

import pandas as pd
import pytest

from pdpipe.basic_stages import ColDrop


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


def test_coldrop_bad_args_in_list():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    with pytest.raises(ValueError):
        stage = ColDrop(['num1', df])
        assert not isinstance(stage, ColDrop)


def test_coldrop_bad_arg_not_str_or_list():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        stage = ColDrop(datetime.datetime.now())
        assert not isinstance(stage, ColDrop)


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
