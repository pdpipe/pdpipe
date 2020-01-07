"""Test the DropTokensByLength pipeline stage."""

import pandas as pd
import pdpipe as pdp


DF = pd.DataFrame(
    data=[[4, ["a", "bad", "nice"]], [5, ["good", "university"]]],
    index=[1, 2],
    columns=["age", "text"],
)


def test_drop_tokens_by_len():
    filt = pdp.DropTokensByLength('text', 3, 5)
    res_df = filt(DF)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert 'bad' in res_df.loc[1]['text']
    assert 'nice' in res_df.loc[1]['text']
    assert 'a' not in res_df.loc[1]['text']
    assert 'good' in res_df.loc[2]['text']
    assert 'university' not in res_df.loc[2]['text']


def test_drop_tokens_by_len_no_max():
    filt = pdp.DropTokensByLength('text', 3)
    res_df = filt(DF)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert 'bad' in res_df.loc[1]['text']
    assert 'nice' in res_df.loc[1]['text']
    assert 'a' not in res_df.loc[1]['text']
    assert 'good' in res_df.loc[2]['text']
    assert 'university' in res_df.loc[2]['text']
