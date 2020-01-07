"""Test the DropTokensByList pipeline stage."""

import pandas as pd
import pdpipe as pdp


def test_drop_tokens_by_list_short():
    data = [[4, ["a", "bad", "cat"]], [5, ["bad", "not", "good"]]]
    df = pd.DataFrame(data, [1, 2], ["age", "text"])
    filter_tokens = pdp.DropTokensByList('text', ['bad'])
    res_df = filter_tokens(df)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert 'bad' not in res_df.loc[1]['text']
    assert 'a' in res_df.loc[1]['text']
    assert 'cat' in res_df.loc[1]['text']
    assert 'bad' not in res_df.loc[2]['text']
    assert 'not' in res_df.loc[2]['text']
    assert 'good' in res_df.loc[2]['text']


def test_drop_tokens_by_list_short_no_drop():
    data = [[4, ["a", "bad", "cat"]], [5, ["bad", "not", "good"]]]
    df = pd.DataFrame(data, [1, 2], ["age", "text"])
    filter_tokens = pdp.DropTokensByList('text', ['bad'], drop=False)
    res_df = filter_tokens(df)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert 'text_filtered' in res_df.columns
    assert 'bad' not in res_df.loc[1]['text_filtered']
    assert 'a' in res_df.loc[1]['text_filtered']
    assert 'cat' in res_df.loc[1]['text_filtered']
    assert 'bad' not in res_df.loc[2]['text_filtered']
    assert 'not' in res_df.loc[2]['text_filtered']
    assert 'good' in res_df.loc[2]['text_filtered']


def test_drop_tokens_by_long_short():
    data = [[4, ["a", "bad", "cat"]], [5, ["bad", "not", "good"]]]
    df = pd.DataFrame(data, [1, 2], ["age", "text"])
    bad = ['z', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'bad']
    filter_tokens = pdp.DropTokensByList('text', bad)
    res_df = filter_tokens(df)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert 'bad' not in res_df.loc[1]['text']
    assert 'a' in res_df.loc[1]['text']
    assert 'cat' in res_df.loc[1]['text']
    assert 'bad' not in res_df.loc[2]['text']
    assert 'not' in res_df.loc[2]['text']
    assert 'good' in res_df.loc[2]['text']
