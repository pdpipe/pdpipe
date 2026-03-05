"""Testing basic pipeline stages."""

import pickle

import pandas as pd

from pdpipe.cq import StartsWith
from pdpipe.basic_stages import ValDrop

from pdptestutil import random_pickle_path


def test_valdrop_with_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ["a", "b"])
    res_df = ValDrop([4], "a").apply(df)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_valdrop_with_columns_verbose():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ["a", "b"])
    res_df = ValDrop([4], "a").apply(df, verbose=True)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_valdrop_without_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ["a", "b"])
    res_df = ValDrop([4]).apply(df)
    assert 1 not in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_valdrop_w_fittable_cq():
    df = pd.DataFrame([[1, 4], [4, 5]], [1, 2], ["aa", "ba"])
    vdrop = ValDrop([4], columns=StartsWith("a"))
    res_df = vdrop(df)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    # now after the column qualifier is fitter, 'ag' should not be transformed
    df = pd.DataFrame([[1, 4], [4, 5]], [1, 2], ["aa", "ag"])
    res_df = vdrop(df)
    assert 1 in res_df.index
    assert 2 not in res_df.index


def test_pickle_valdrop(pdpipe_tests_dir_path):
    """Testing ValDrop pickling."""
    df = pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ["a", "b"])
    stage = ValDrop([4], "a")
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage.apply(df)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index
