"""Test nltk pipeline stages."""

import pytest
import pandas as pd
import pdpipe as pdp


@pytest.fixture(scope="session", autouse=True)
def test_tokenize():
    df = pd.DataFrame([[3.2, "Kick the baby!"]], [1], ['freq', 'content'])
    tokenize_stage = pdp.TokenizeWords('content')
    res_df = tokenize_stage(df)
    assert 'content' in res_df.columns
    assert 'content_tok' not in res_df.columns
    assert res_df['content'][1] == ['Kick', 'the', 'baby', '!']

    tokenize_stage = pdp.TokenizeWords('content', drop=False)
    res_df = tokenize_stage(df)
    assert 'content' in res_df.columns
    assert 'content_tok' in res_df.columns
    assert res_df['content'][1] == "Kick the baby!"
    assert res_df['content_tok'][1] == ['Kick', 'the', 'baby', '!']


@pytest.fixture(scope="session", autouse=True)
def test_untokenize():
    df = pd.DataFrame([[3.2, ['Shake', 'and', 'bake!']]], [1], ['freq', 'txt'])
    untok = pdp.UntokenizeWords('txt')
    res_df = untok(df)
    assert 'txt' in res_df.columns
    assert 'txt_untok' not in res_df.columns
    assert res_df['txt'][1] == "Shake and bake!"

    untok = pdp.UntokenizeWords('txt', drop=False)
    res_df = untok(df)
    assert 'txt' in res_df.columns
    assert 'txt_untok' in res_df.columns
    assert res_df['txt'][1] == ['Shake', 'and', 'bake!']
    assert res_df['txt_untok'][1] == "Shake and bake!"


@pytest.fixture(scope="session", autouse=True)
def test_remove_stopwords():
    df = pd.DataFrame([[3.2, ['kick', 'the', 'baby']]], [1], ['freq', 'txt'])
    stop = pdp.RemoveStopwords('txt')
    res_df = stop(df)
    assert 'txt' in res_df.columns
    assert 'txt_nostop' not in res_df.columns
    assert res_df['txt'][1] == ['kick', 'baby']

    untok = pdp.RemoveStopwords('txt', drop=False)
    res_df = untok(df)
    assert 'txt' in res_df.columns
    assert 'txt_nostop' in res_df.columns
    assert res_df['txt'][1] == ['kick', 'the', 'baby']
    assert res_df['txt_untok'][1] == ['kick', 'baby']


@pytest.fixture(scope="session", autouse=True)
def test_snowball_stem():
    df = pd.DataFrame([[3.2, ['kicking', 'boats']]], [1], ['freq', 'txt'])
    stem = pdp.SnowballStem('txt')
    res_df = stem(df)
    assert 'txt' in res_df.columns
    assert 'txt_stem' not in res_df.columns
    assert res_df['txt'][1] == ['kick', 'boat']

    stem = pdp.RemoveStopwords('txt', drop=False)
    res_df = stem(df)
    assert 'txt' in res_df.columns
    assert 'txt_stem' in res_df.columns
    assert res_df['txt'][1] == ['kicking', 'boats']
    assert res_df['txt_stem'][1] == ['kick', 'boat']
