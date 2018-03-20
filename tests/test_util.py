"""Testing pdpipe util module."""

import pandas as pd
import pytest

from pdpipe.util import out_of_place_col_insert


def _test_df():
    return pd.DataFrame(
        data=[[1, 'a'], [2, 'b']],
        index=[1, 2],
        columns=['num', 'char']
    )


def test_out_of_place_col_insert_all_params():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    series = pd.Series(
        data=[10, 20],
        index=[1, 2],
        name='tens')

    result_df = out_of_place_col_insert(df, series, 1, 'Tigers')
    assert 'tens' not in result_df.columns
    assert 'Tigers' in result_df.columns
    assert result_df.columns.get_loc('Tigers') == 1
    assert result_df['Tigers'][1] == 10
    assert result_df['Tigers'][2] == 20


def test_out_of_place_col_insert_no_col_name():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    series = pd.Series(
        data=[10, 20],
        index=[1, 2],
        name='tens')

    result_df = out_of_place_col_insert(df, series, 1)
    assert 'tens' in result_df.columns
    assert result_df.columns.get_loc('tens') == 1
    assert result_df['tens'][1] == 10
    assert result_df['tens'][2] == 20


def test_out_of_place_col_insert_nameless_error():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    series = pd.Series(
        data=[10, 20],
        index=[1, 2])

    with pytest.raises(ValueError):
        out_of_place_col_insert(df, series, 1)


def test_out_of_place_col_last_position():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()
    series = pd.Series(
        data=[10, 20],
        index=[1, 2],
        name='tens')

    result_df = out_of_place_col_insert(df, series, len(df.columns), 'Tigers')
    assert 'tens' not in result_df.columns
    assert 'Tigers' in result_df.columns
    assert result_df.columns.get_loc('Tigers') == 2
    assert result_df['Tigers'][1] == 10
    assert result_df['Tigers'][2] == 20
