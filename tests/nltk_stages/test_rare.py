"""Test DropRareTokens stages."""

import pytest
import pandas as pd

from pdpipe.nltk_stages import DropRareTokens


def _some_df():
    return pd.DataFrame(
        data=[[7, ['a', 'a', 'b']], [3, ['b', 'c', 'd']]],
        index=[1, 2],
        columns=["num", "chars"]
    )


def _some_df2():
    return pd.DataFrame(
        data=[[7, ['a', 'c', 'c']], [3, ['b', 'd', 'd']]],
        index=[1, 2],
        columns=["num", "chars"]
    )


@pytest.mark.first
def test_drop_rare():
    df = _some_df()
    drop_rare_stage = DropRareTokens('chars', 1)
    res_df = drop_rare_stage(df)
    assert 'chars' in res_df.columns
    assert res_df['chars'][1] == ['a', 'a', 'b']
    assert res_df['chars'][2] == ['b']

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = drop_rare_stage(df2)
    assert 'chars' in res_df2.columns
    assert res_df2['chars'][1] == ['a']
    assert res_df2['chars'][2] == ['b']

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = drop_rare_stage.fit_transform(df2)
    assert 'chars' in res_df2.columns
    assert res_df2['chars'][1] == ['c', 'c']
    assert res_df2['chars'][2] == ['d', 'd']


@pytest.mark.first
def test_drop_rare_w_drop():
    df = _some_df()
    drop_rare_stage = DropRareTokens('chars', 1, drop=False)
    res_df = drop_rare_stage(df, verbose=True)
    assert 'chars' in res_df.columns
    assert 'chars_norare' in res_df.columns
    assert res_df['chars_norare'][1] == ['a', 'a', 'b']
    assert res_df['chars_norare'][2] == ['b']
    assert res_df['chars'][2] == ['b', 'c', 'd']

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = drop_rare_stage(df2, verbose=True)
    assert 'chars' in res_df2.columns
    assert 'chars_norare' in res_df2.columns
    assert res_df2['chars_norare'][1] == ['a']
    assert res_df2['chars_norare'][2] == ['b']
    assert res_df2['chars'][1] == ['a', 'c', 'c']
    assert res_df2['chars'][2] == ['b', 'd', 'd']

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = drop_rare_stage.fit_transform(df2)
    assert 'chars' in res_df2.columns
    assert 'chars_norare' in res_df2.columns
    assert res_df2['chars_norare'][1] == ['c', 'c']
    assert res_df2['chars_norare'][2] == ['d', 'd']
    assert res_df2['chars'][1] == ['a', 'c', 'c']
    assert res_df2['chars'][2] == ['b', 'd', 'd']
