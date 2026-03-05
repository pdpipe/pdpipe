"""Testing ColRename pipeline stages."""

import pickle

import pandas as pd

from pdpipe.basic_stages import SetIndex

from pdptestutil import random_pickle_path


def test_set_index():
    """Testing the SetIndex pipeline stage."""
    df = pd.DataFrame(
        data=[[8, "a", 4], [5, "b", 92]],
        index=[1, 2],
        columns=["num", "char", "w"],
    )
    res_df = SetIndex("num", drop=True).apply(df)
    assert "w" in res_df.columns
    assert "num" not in res_df.columns
    assert "char" in res_df.columns
    assert res_df.index.name == "num"

    res_df = SetIndex(["w", "num"], drop=True).apply(df)
    assert "w" not in res_df.columns
    assert "num" not in res_df.columns
    assert "char" in res_df.columns


def test_pickle_set_index(pdpipe_tests_dir_path):
    """Testing SetIndex pickling."""
    df = pd.DataFrame(
        data=[[8, "a", 4], [5, "b", 92]],
        index=[1, 2],
        columns=["num", "char", "w"],
    )
    stage = SetIndex("num", drop=True)
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage.apply(df)
    assert "w" in res_df.columns
    assert "num" not in res_df.columns
    assert "char" in res_df.columns
    assert res_df.index.name == "num"
