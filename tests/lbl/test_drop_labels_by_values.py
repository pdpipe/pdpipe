"""Testing theDropLabelsByValues pipeline stage."""

from typing import Union, List

import pytest
import pandas as pd
import pdpipe as pdp
from pdpipe import df
from pdpipe.lbl import DropLabelsByValues
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator
from sklearn.linear_model import LogisticRegression


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

X2 = pd.DataFrame(
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


def _equal(
    first: Union[pd.Series, pd.Index],
    second: Union[List, pd.Index],
) -> bool:
    if isinstance(first, pd.Series):
        return first.values.tolist() == second
    else:
        return first.tolist() == second.tolist()


@pytest.mark.label
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


class MyPipelineAndModel(PdPipelineAndSklearnEstimator):

    def __init__(self, in_set, skip=False):
        if skip:
            drop_stage = DropLabelsByValues(in_set=in_set)
        else:
            drop_stage = DropLabelsByValues(in_set=in_set)
        pipeline = pdp.PdPipeline(stages=[
            df['hage'] << df['Age'] / 2,
            drop_stage,
            pdp.ColDrop('Name'),
        ])
        model = LogisticRegression()
        super().__init__(pipeline=pipeline, estimator=model)


@pytest.mark.label
def test_drop_labels_by_values_with_label_placeholder_predict():
    pmodel = MyPipelineAndModel(in_set=[1, 3])
    pmodel.fit(X1, Y1)
    x, y = pmodel.pipeline.transform(X1, Y1)
    assert x.shape == (6, 2)
    assert len(y) == 6
    assert _equal(y, [0, 0, 0, 2, 2, 4])
    expected_index = pd.Index([1, 2, 3, 6, 7, 10])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)

    pred_y2 = pmodel.predict(X2)
    # this makes sure that the DropLabelsByValues stage is not applied to
    # input dataframes on prediciton
    assert pred_y2.shape == (10, )


@pytest.mark.label
def test_drop_labels_skip_is_ignored():
    pmodel = MyPipelineAndModel(in_set=[1, 3])
    pmodel.fit(X1, Y1)
    x, y = pmodel.pipeline.transform(X1, Y1)
    assert x.shape == (6, 2)
    assert len(y) == 6
    assert _equal(y, [0, 0, 0, 2, 2, 4])
    expected_index = pd.Index([1, 2, 3, 6, 7, 10])
    assert _equal(y.index, expected_index)
    assert _equal(x.index, expected_index)

    pred_y2 = pmodel.predict(X2)
    # this makes sure that the DropLabelsByValues stage is not applied to
    # input dataframes on prediciton
    assert pred_y2.shape == (10, )
