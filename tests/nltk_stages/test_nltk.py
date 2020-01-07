"""Test nltk pipeline stages."""

import os
import sys

import pytest
import pandas as pd
import pdpipe as pdp


@pytest.mark.first
def test_tokenize():
    df = pd.DataFrame([[3.2, "Kick the baby!"]], [1], ['freq', 'content'])
    tokenize_stage = pdp.TokenizeText('content')
    res_df = tokenize_stage(df)
    assert 'content' in res_df.columns
    assert 'content_tok' not in res_df.columns
    assert res_df['content'][1] == ['Kick', 'the', 'baby', '!']

    tokenize_stage = pdp.TokenizeText('content', drop=False)
    res_df = tokenize_stage(df)
    assert 'content' in res_df.columns
    assert 'content_tok' in res_df.columns
    assert res_df['content'][1] == "Kick the baby!"
    assert res_df['content_tok'][1] == ['Kick', 'the', 'baby', '!']


@pytest.mark.first
def test_untokenize():
    df = pd.DataFrame([[3.2, ['Shake', 'and', 'bake!']]], [1], ['freq', 'txt'])
    untok = pdp.UntokenizeText('txt')
    res_df = untok(df)
    assert 'txt' in res_df.columns
    assert 'txt_untok' not in res_df.columns
    assert res_df['txt'][1] == "Shake and bake!"

    untok = pdp.UntokenizeText('txt', drop=False)
    res_df = untok(df)
    assert 'txt' in res_df.columns
    assert 'txt_untok' in res_df.columns
    assert res_df['txt'][1] == ['Shake', 'and', 'bake!']
    assert res_df['txt_untok'][1] == "Shake and bake!"


@pytest.mark.first
@pytest.mark.skipif(
    (os.name == 'nt') or (sys.platform.startswith('win')),
    reason="nltk has a problem locating resources on windows",
)
def test_remove_stopwords():
    df = pd.DataFrame([[3.2, ['kick', 'the', 'baby']]], [1], ['freq', 'txt'])
    stop = pdp.RemoveStopwords('english', 'txt')
    res_df = stop(df)
    assert 'txt' in res_df.columns
    assert 'txt_nostop' not in res_df.columns
    assert res_df['txt'][1] == ['kick', 'baby']

    stop = pdp.RemoveStopwords('english', 'txt', drop=False)
    res_df = stop(df)
    assert 'txt' in res_df.columns
    assert 'txt_nostop' in res_df.columns
    assert res_df['txt'][1] == ['kick', 'the', 'baby']
    assert res_df['txt_nostop'][1] == ['kick', 'baby']

    stop = pdp.RemoveStopwords(['baby', 'fart'], 'txt', drop=False)
    res_df = stop(df)
    assert 'txt' in res_df.columns
    assert 'txt_nostop' in res_df.columns
    assert res_df['txt'][1] == ['kick', 'the', 'baby']
    assert res_df['txt_nostop'][1] == ['kick', 'the']

    with pytest.raises(TypeError):
        pdp.RemoveStopwords(34, 'txt')
