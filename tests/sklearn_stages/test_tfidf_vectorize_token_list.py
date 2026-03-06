"""Test the TfidfVectorizeTokenLists pipeline stage."""

import pickle

import pandas as pd
import pytest

import pdpipe as pdp

from pdptestutil import random_pickle_path

DF = pd.DataFrame(
    data=[
        [23, ["live", "full", "cats", "mango"]],
        [80, ["hovercraft", "full", "eels"]],
    ],
    columns=["Age", "Quote"],
)


DF2 = pd.DataFrame(
    data=[
        [29, ["eels", "full", "banana"]],
    ],
    columns=["Age", "Quote"],
)


@pytest.mark.parametrize("drop", [True, False])
def test_tfidf_vec(drop):
    tf = pdp.TfidfVectorizeTokenLists("Quote", drop=drop)
    res_df = tf(DF)
    assert "Age" in res_df.columns
    words = []
    for i, row in DF.iterrows():
        words.extend(row["Quote"])
    uwords = set(words)
    for w in uwords:
        assert w in res_df.columns
    non_zeros = (
        (res_df.drop("Quote", axis=1, errors="ignore") > 0).T.sum().values
    )
    for i, row in DF.iterrows():
        assert len(row["Quote"]) == non_zeros[i] - 1
    if not drop:
        assert "Quote" in res_df.columns

    res_df2 = tf(DF2)
    assert "Age" in res_df2.columns
    if not drop:
        assert "Quote" in res_df2.columns
    non_zeros2 = (
        (res_df2.drop("Quote", axis=1, errors="ignore") > 0).T.sum().values
    )
    assert non_zeros2[0] - 1 == 2


def test_tfidf_vec_hierarchical_labels():
    tf = pdp.TfidfVectorizeTokenLists("Quote", hierarchical_labels=True)
    res_df = tf(DF)
    assert "Age" in res_df.columns
    words = []
    for i, row in DF.iterrows():
        words.extend(row["Quote"])
    uwords = set(words)
    for w in uwords:
        assert f"Quote_{w}" in res_df.columns
    non_zeros = (
        (res_df.drop("Quote", axis=1, errors="ignore") > 0).T.sum().values
    )
    for i, row in DF.iterrows():
        assert len(row["Quote"]) == non_zeros[i] - 1


def test_pickle_tfidf_vec(pdpipe_tests_dir_path):
    """Testing TfidfVectorizeTokenLists pickling."""
    tf = pdp.TfidfVectorizeTokenLists("Quote")
    tf(DF)
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(tf, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df2 = loaded_stage(DF2)
    assert "Age" in res_df2.columns


def test_sklearn_missing_dep_tfidf():
    """Test TfidfVectorizeTokenLists raises ImportError without sklearn."""
    import pdpipe.sklearn_stages as sk

    original = sk._SKLEARN_INSTALLED
    try:
        sk._SKLEARN_INSTALLED = False
        with pytest.raises(ImportError, match="scikit-learn is required"):
            pdp.TfidfVectorizeTokenLists("tokens")
    finally:
        sk._SKLEARN_INSTALLED = original
