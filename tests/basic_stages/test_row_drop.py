"""Testing basic pipeline stages."""

import pytest
import pandas as pd

from pdpipe.basic_stages import RowDrop

DF1 = pd.DataFrame(
    data=[[1, 4], [4, 5], [18, 11]],
    index=[1, 2, 3],
    columns=['a', 'b'],
)

DF2 = pd.DataFrame(
    data=[[1, 4, 3], [4, 5, -3], [18, 11, 9]],
    index=[1, 2, 3],
    columns=['a', 'b', 'c'],
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

    res_df = RowDrop([lambda x: x < 2], columns=['a', 'b']).apply(DF2)
    assert 1 not in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_row_drop_bad_reducer():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        RowDrop([lambda x: x < 2], reduce='al')


def test_row_drop_bad_condition_in_dict():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        RowDrop({'a': 'bad'})


def test_row_drop_bad_condition_in_list():
    """Testing the ColDrop pipeline stage."""
    with pytest.raises(ValueError):
        RowDrop(['bad'])
