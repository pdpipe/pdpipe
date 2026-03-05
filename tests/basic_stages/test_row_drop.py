"""Testing basic pipeline stages."""

import pickle

import pytest
import pandas as pd

from pdpipe.basic_stages import RowDrop
from pdpipe.exceptions import FailedPreconditionError

from pdptestutil import random_pickle_path

DF1 = pd.DataFrame(
    data=[[1, 4], [4, 5], [18, 11]],
    index=[1, 2, 3],
    columns=["a", "b"],
)

DF2 = pd.DataFrame(
    data=[[1, 4, 3], [4, 5, -3], [18, 11, 9]],
    index=[1, 2, 3],
    columns=["a", "b", "c"],
)

DF3 = pd.DataFrame(
    data=[[1, 2], [0, 5], [18, 11]],
    index=[1, 2, 3],
    columns=["a", "b"],
)


def test_row_drop():
    """Testing the ColDrop pipeline stage."""
    res_df = RowDrop([lambda x: x < 2]).apply(DF1)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_row_drop_verbose():
    """Testing the ColDrop pipeline stage."""
    res_df = RowDrop([lambda x: x < 2]).apply(DF1, verbose=True)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_row_drop_columns():
    """Testing the ColDrop pipeline stage."""
    res_df = RowDrop([lambda x: x < 2]).apply(DF2)
    assert 1 not in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index

    res_df = RowDrop([lambda x: x < 2], columns=["a", "b"]).apply(DF2)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_row_drop_bad_reducer():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        RowDrop([lambda x: x < 2], reduce="al")


def test_row_drop_bad_condition_in_dict():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        RowDrop({"a": "bad"})


def test_row_drop_bad_condition_in_list():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        RowDrop(["bad"])


def test_row_drop_all_reducer():
    """Testing the ColDrop pipeline stage."""
    res_df = RowDrop([lambda x: x < 3]).apply(DF3)
    assert 1 not in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index

    res_df = RowDrop([lambda x: x < 3], reduce="all").apply(DF3)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_row_drop_bad_columns():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(FailedPreconditionError):
        RowDrop([lambda x: x < 2], columns=["d"]).apply(DF1)


def test_row_drop_xor_reducer():
    """Testing the ColDrop pipeline stage."""
    res_df = RowDrop([lambda x: x < 3]).apply(DF3)
    assert 1 not in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index

    res_df = RowDrop([lambda x: x < 3], reduce="xor").apply(DF3)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def _value_less_than_2(x):
    return x < 2


def test_pickle_rowdrop(pdpipe_tests_dir_path):
    """Testing RowDrop pickling."""
    stage = RowDrop([_value_less_than_2])
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage.apply(DF1)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index
