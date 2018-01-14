"""Testing basic pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import ValDrop


def test_valdrop_with_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ['a', 'b'])
    res_df = ValDrop([4], 'a').apply(df)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_valdrop_with_columns_verbose():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ['a', 'b'])
    res_df = ValDrop([4], 'a').apply(df, verbose=True)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_valdrop_without_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ['a', 'b'])
    res_df = ValDrop([4]).apply(df)
    assert 1 not in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index
