"""Testing basic pipline stages."""

import pandas as pd

from pdpipe.basic_stages import Binarize


def _one_categ_df():
    return pd.DataFrame([['USA'], ['UK'], ['Greece']], [1, 2, 3], ['Born'])


def test_binarize_one():
    """Basic binning test."""
    df = _one_categ_df()
    res_df = Binarize('Born').apply(df)
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
    assert 'Name' in res_df.columns
    assert 'Name_Bon' not in res_df.columns
    assert 'Born_UK' in res_df.columns
    assert res_df['Born_UK'][1] == 0
    assert res_df['Born_UK'][2] == 1
    assert res_df['Born_UK'][3] == 0
    assert 'Born_USA' in res_df.columns
    assert res_df['Born_USA'][1] == 1
    assert res_df['Born_USA'][2] == 0
    assert res_df['Born_USA'][3] == 0
