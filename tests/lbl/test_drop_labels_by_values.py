"""Testing theDropLabelsByValues pipeline stage."""

from typing import Union, List

import pandas as pd

from pdpipe.lbl import DropLabelsByValues


X1 = pd.DataFrame(
    data=[
        [23, 'Jo'],
        [52, 'Regina'],
        [23, 'Dana'],
        [25, 'Bo'],
        [80, 'Richy'],
        [60, 'Paul'],
        [44, 'Derek'],
        [72, 'Regina'],
        [50, 'Jim'],
        [80, 'Wealthus'],
    ],
    columns=['Age', 'Name'],
    index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)


Y1 = pd.Series(
    data=[0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
    index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)


def _equal(
    first: Union[pd.Series, pd.Index],
    second: Union[List, pd.Index],
) -> bool:
    if isinstance(first, pd.Series):
        return first.values.tolist() == second
    else:
        return first.tolist() == second.tolist()


def test_drop_labels_by_values():
    # test drop_labels_by_values with in_set = [1,3]
    stage = DropLabelsByValues(in_set=[1, 3])
    x, y = stage(X1, Y1)
    assert x.shape == (6, 2)
    assert len(y) == 6
    assert _equal(y, [0, 0, 0, 2, 2, 4])
    expected_index = pd.Index([1, 2, 3, 6, 7, 10])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)

    # test drop_labels_by_values with in_ranges = [(1,3)]]
    stage = DropLabelsByValues(in_ranges=[(1, 3)])
    x, y = stage(X1, Y1)
    assert x.shape == (4, 2)
    assert len(y) == 4
    assert _equal(y, [0, 0, 0, 4])
    expected_index = pd.Index([1, 2, 3, 10])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)

    # test drop_labels_by_values with in_ranges = [(0,2),(4,7)]
    stage = DropLabelsByValues(in_ranges=[(0, 2), (4, 7)])
    x, y = stage(X1, Y1)
    assert x.shape == (2, 2)
    assert len(y) == 2
    assert _equal(y, [3, 3])
    expected_index = pd.Index([8, 9])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)

    # test drop_labels_by_values with not_in_set = [1,3]
    stage = DropLabelsByValues(not_in_set=[1, 3])
    x, y = stage(X1, Y1)
    assert x.shape == (4, 2)
    assert len(y) == 4
    assert _equal(y, [1, 1, 3, 3])
    expected_index = pd.Index([4, 5, 8, 9])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)

    # test drop_labels_by_values with not_in_ranges = [(1,3)]]
    stage = DropLabelsByValues(not_in_ranges=[(1, 3)])
    x, y = stage(X1, Y1)
    assert x.shape == (6, 2)
    assert len(y) == 6
    assert _equal(y, [1, 1, 2, 2, 3, 3])
    expected_index = pd.Index([4, 5, 6, 7, 8, 9])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)
