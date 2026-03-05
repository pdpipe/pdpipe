"""Testing basic pipeline stages."""

import pickle

import pandas as pd
import pytest

from pdpipe import DropDuplicates
from pdpipe.exceptions import FailedPreconditionError

from pdptestutil import random_pickle_path


def _test_df():
    return pd.DataFrame([[8, 1], [8, 2], [9, 2]], [1, 2, 3], ["a", "b"])


def test_drop_duplicates():
    df = _test_df()
    stage = DropDuplicates("a")
    res = stage.apply(df)
    assert 1 in res.index
    assert 2 not in res.index
    assert 3 in res.index

    stage = DropDuplicates("b")
    res = stage.apply(df)
    assert 1 in res.index
    assert 2 in res.index
    assert 3 not in res.index

    stage = DropDuplicates(["a", "b"])
    res = stage.apply(df)
    assert 1 in res.index
    assert 2 in res.index
    assert 3 in res.index

    stage = DropDuplicates()
    res = stage.apply(df, verbose=True)
    assert 1 in res.index
    assert 2 in res.index
    assert 3 in res.index

    stage = DropDuplicates("c")
    with pytest.raises(FailedPreconditionError):
        stage.apply(df)


def test_pickle_drop_duplicates(pdpipe_tests_dir_path):
    """Testing DropDuplicates pickling."""
    df = _test_df()
    stage = DropDuplicates("a")
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res = loaded_stage.apply(df)
    assert 1 in res.index
    assert 2 not in res.index
    assert 3 in res.index
