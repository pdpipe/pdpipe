"""Testing basic pipeline stages."""

import pickle

import pandas as pd

from pdpipe.basic_stages import DropNa

from pdptestutil import random_pickle_path

DF1 = pd.DataFrame([[1, 4], [4, None], [1, 11]], [1, 2, 3], ["a", "b"])


def test_dropna_basic():
    """Testing the DropNa pipeline stage."""
    res_df = DropNa().apply(DF1)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_dropna_verbose():
    """Testing the DropNa pipeline stage."""
    res_df = DropNa().apply(DF1, verbose=True)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_pickle_dropna(pdpipe_tests_dir_path):
    """Testing DropNa pickling."""
    stage = DropNa()
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage.apply(DF1)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index
