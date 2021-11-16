"""Test the RegexReplace pipeline stage."""

import re

import pandas as pd
import pdpipe as pdp


DF = pd.DataFrame(
    data=[[4, "more than 12"], [5, "with 5 more"]],
    index=[1, 2],
    columns=["age", "text"],
)


def test_regex_replace():
    clean_num = pdp.RegexReplace('text', r'\b[0-9]+\b', "NUM")
    res_df = clean_num(DF)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert res_df.loc[1]['text'] == 'more than NUM'
    assert res_df.loc[2]['text'] == 'with NUM more'


def test_regex_replace_no_drop():
    clean_num = pdp.RegexReplace('text', r'\b[0-9]+\b', "NUM", drop=False)
    res_df = clean_num(DF)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert 'text_regex' in res_df.columns
    assert res_df.loc[1]['text'] == 'more than 12'
    assert res_df.loc[2]['text'] == 'with 5 more'
    assert res_df.loc[1]['text_regex'] == 'more than NUM'
    assert res_df.loc[2]['text_regex'] == 'with NUM more'


DF2 = pd.DataFrame(
    data=[[4, "first\nsecond"], [5, "1 \n 2 \n 3"]],
    index=[1, 2],
    columns=["age", "text"],
)


DF3 = pd.DataFrame(
    data=[["Mr. John", 18], ["MR. Bob", 25]],
    index=[1, 2],
    columns=["name", "age"],
)


def test_regex_replace_with_flags():
    tokenizer = pdp.RegexReplace('text', r'.+', "TOKEN")
    res_df = tokenizer(DF2)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert res_df.loc[1]['text'] == 'TOKEN\nTOKEN'
    assert res_df.loc[2]['text'] == 'TOKEN\nTOKEN\nTOKEN'

    # now with the DOTALL flag
    tokenizer = pdp.RegexReplace('text', r'.+', "TOKEN", flags=re.DOTALL)
    res_df = tokenizer(DF2)
    assert 'age' in res_df.columns
    assert 'text' in res_df.columns
    assert res_df.loc[1]['text'] == 'TOKEN'
    assert res_df.loc[2]['text'] == 'TOKEN'

    # documentation example
    tokenizer = pdp.RegexReplace('name', r'^mr.*', "x", flags=re.IGNORECASE)
    res_df = tokenizer(DF3)
    assert 'name' in res_df.columns
    assert 'age' in res_df.columns
    assert res_df.loc[1]['name'] == 'x'
    assert res_df.loc[2]['name'] == 'x'
