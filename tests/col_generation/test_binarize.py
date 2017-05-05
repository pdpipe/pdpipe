"""Testing Binarize pipline stages."""

import pandas as pd

from pdpipe.col_generation import Binarize


def _one_categ_df():
    return pd.DataFrame([['USA'], ['UK'], ['Greece']], [1, 2, 3], ['Born'])


def test_binarize_one():
    """Basic binning test."""
    df = _one_categ_df()
    res_df = Binarize('Born').apply(df, verbose=True)
    assert 'Born' not in res_df.columns
    assert 'Greece' not in res_df.columns
    assert 'Born_UK' in res_df.columns
    assert res_df['Born_UK'][1] == 0
    assert res_df['Born_UK'][2] == 1
    assert res_df['Born_UK'][3] == 0
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0


def test_binarize_no_drop_first():
    """Basic binning test."""
    df = _one_categ_df()
    res_df = Binarize('Born', drop_first=False).apply(df)
    assert 'Born' not in res_df.columns
    assert 'Born_UK' in res_df.columns
    assert res_df['Born_UK'][1] == 0
    assert res_df['Born_UK'][2] == 1
    assert res_df['Born_UK'][3] == 0
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0
    assert 'Born_Greece' in res_df.columns
    assert res_df['Born_Greece'][1] == 0
    assert res_df['Born_Greece'][2] == 0
    assert res_df['Born_Greece'][3] == 1


def _two_categ_df():
    return pd.DataFrame(
        data=[['USA', 'Bob'], ['UK', 'Jack'], ['Greece', 'Yan']],
        index=[1, 2, 3],
        columns=['Born', 'Name']
    )


def test_binarize_one_with_exclude():
    """Basic binning test."""
    df = _two_categ_df()
    res_df = Binarize(exclude_columns=['Name']).apply(df)
    assert 'Born' not in res_df.columns
    assert 'Name' in res_df.columns
    assert 'Name_Bob' not in res_df.columns
    assert 'Name_Jack' not in res_df.columns
    assert 'Name_Yan' not in res_df.columns
    assert 'Greece' not in res_df.columns
    assert 'Born_UK' in res_df.columns
    assert res_df['Born_UK'][1] == 0
    assert res_df['Born_UK'][2] == 1
    assert res_df['Born_UK'][3] == 0
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0


def _one_categ_df_with_nan():
    return pd.DataFrame([['USA'], ['UK'], [None]], [1, 2, 3], ['Born'])


def test_binarize_with_nan():
    """Basic binning test."""
    df = _one_categ_df_with_nan()
    res_df = Binarize('Born').apply(df)
    assert 'Born' not in res_df.columns
    assert 'Born_UK' not in res_df.columns
    assert 'Born_nan' not in res_df.columns
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0


def test_binarize_with_dummy_na():
    """Basic binning test."""
    df = _one_categ_df_with_nan()
    res_df = Binarize('Born', dummy_na=True).apply(df)
    assert 'Born' not in res_df.columns
    assert 'Born_nan' not in res_df.columns
    assert 'Born_UK' in res_df.columns
    assert res_df['Born_UK'][1] == 0
    assert res_df['Born_UK'][2] == 1
    assert res_df['Born_UK'][3] == 0
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0


def test_binarize_one_no_drop():
    """Basic binning test."""
    df = _one_categ_df()
    res_df = Binarize('Born', drop=False).apply(df, verbose=True)
    assert 'Greece' not in res_df.columns
    assert 'Born' in res_df.columns
    assert 'Born_UK' in res_df.columns
    assert res_df['Born_UK'][1] == 0
    assert res_df['Born_UK'][2] == 1
    assert res_df['Born_UK'][3] == 0
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0
