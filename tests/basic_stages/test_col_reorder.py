"""Testing the ColReorder stage."""

import pickle

import pytest
import pandas as pd

from pdpipe import ColReorder
from pdpipe.exceptions import FailedPreconditionError

from pdptestutil import random_pickle_path


def _df():
    return pd.DataFrame([[8, 4, 3, 7]], columns=["a", "b", "c", "d"])


def test_colreorder():
    df = _df()
    stage = ColReorder({"b": 0, "c": 3})
    res = stage(df)
    assert list(res.columns) == ["b", "a", "d", "c"]

    stage2 = ColReorder({"g": 0})
    with pytest.raises(FailedPreconditionError):
        stage2(df)

    stage3 = ColReorder({"a": 8})
    with pytest.raises(ValueError):
        stage3(df)


def test_pickle_colreorder(pdpipe_tests_dir_path):
    """Testing ColReorder pickling."""
    df = _df()
    stage = ColReorder({"b": 0, "c": 3})
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res = loaded_stage(df)
    assert list(res.columns) == ["b", "a", "d", "c"]
