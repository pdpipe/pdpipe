"""Test the SnowballStem pipeline stage."""

import pickle

import pytest
import pandas as pd
import pdpipe as pdp

from pdptestutil import random_pickle_path


@pytest.mark.first
def test_snowball_stem():
    df = pd.DataFrame([[3.2, ["kicking", "boats"]]], [1], ["freq", "txt"])
    stem = pdp.SnowballStem("EnglishStemmer", "txt")
    res_df = stem(df)
    assert "txt" in res_df.columns
    assert "txt_stem" not in res_df.columns
    assert res_df["txt"][1] == ["kick", "boat"]

    stem = pdp.SnowballStem("EnglishStemmer", "txt", drop=False)
    res_df = stem(df)
    assert "txt" in res_df.columns
    assert "txt_stem" in res_df.columns
    assert res_df["txt"][1] == ["kicking", "boats"]
    assert res_df["txt_stem"][1] == ["kick", "boat"]


@pytest.mark.first
def test_snowball_stem_cond():
    df = pd.DataFrame([[3.2, ["kicking", "boats"]]], [1], ["freq", "txt"])
    stem = pdp.SnowballStem("EnglishStemmer", "txt", min_len=7)
    res_df = stem(df)
    assert "txt" in res_df.columns
    assert "txt_stem" not in res_df.columns
    assert res_df["txt"][1] == ["kick", "boats"]

    stem = pdp.SnowballStem("EnglishStemmer", "txt", min_len=5)
    res_df = stem(df)
    assert "txt" in res_df.columns
    assert "txt_stem" not in res_df.columns
    assert res_df["txt"][1] == ["kick", "boat"]

    stem = pdp.SnowballStem("EnglishStemmer", "txt", max_len=5)
    res_df = stem(df)
    assert "txt" in res_df.columns
    assert "txt_stem" not in res_df.columns
    assert res_df["txt"][1] == ["kicking", "boat"]

    df = pd.DataFrame(
        data=[[3.2, ["boats", "kicking", "squealing"]]],
        index=[1],
        columns=["freq", "txt"],
    )
    stem = pdp.SnowballStem("EnglishStemmer", "txt", min_len=6, max_len=7)
    res_df = stem(df)
    assert "txt" in res_df.columns
    assert "txt_stem" not in res_df.columns
    assert res_df["txt"][1] == ["boats", "kick", "squealing"]


@pytest.mark.first
def test_pickle_snowball_stem(pdpipe_tests_dir_path):
    """Testing SnowballStem pickling."""
    df = pd.DataFrame([[3.2, ["kicking", "boats"]]], [1], ["freq", "txt"])
    stage = pdp.SnowballStem("EnglishStemmer", "txt")
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage(df)
    assert "txt" in res_df.columns
    assert res_df["txt"][1] == ["kick", "boat"]
