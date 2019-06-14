"""Testing ValKeep pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import ValKeep


def test_valkeep_with_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [5, 11]], [1, 2, 3], ['a', 'b'])
    res_df = ValKeep([4, 5], 'a').apply(df)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_valkeep_with_columns_verbose():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [5, 11]], [1, 2, 3], ['a', 'b'])
    res_df = ValKeep([4, 5], 'a').apply(df, verbose=True)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_valkeep_without_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [5, 11]], [1, 2, 3], ['a', 'b'])
    res_df = ValKeep([4, 5]).apply(df)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 not in res_df.index
