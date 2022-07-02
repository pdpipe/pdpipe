"""Testing sklearn_stages.LabelEncoder."""

import math

import pytest
import numpy as np
import pandas as pd

import pdpipe as pdp
from pdpipe.sklearn_stages import EncodeLabel
from pdpipe.exceptions import UnfittedPipelineStageError


def _some_X_y():
    X = pd.DataFrame(
        data=[[3.2, 31], [7.2, 33], [12.1, 28]],
        index=[1, 2, 3],
        columns=["ph", "temp"]
    )
    y = pd.Series(["acd", "alk", "alk"])
    return X, y


def _some_X_y2():
    X = pd.DataFrame(
        data=[[4.1, 22], [7.7, 23], [2.1, 38]],
        index=[1, 2, 3],
        columns=["ph", "temp"]
    )
    y = pd.Series(["alk", "acd", "alk"])
    return X, y


def test_encode_label():
    X, y = _some_X_y()
    encode_stage = EncodeLabel()

    post_X, post_y = encode_stage(X, y)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 0
    assert post_y[2] == 1
    assert post_y[3] == 1

    # see only transform (no fit) when already fitted
    X2, y2 = _some_X_y2()
    post_X, post_y = encode_stage(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1

    # see only transform (no fit) when already fitted
    X2, y2 = _some_X_y2()
    post_X, post_y = encode_stage.transform(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1

    # check fit_transform when already fitted
    X2, y2 = _some_X_y2()
    post_X, post_y = encode_stage.fit_transform(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1


def test_encode_label_skip():
    X, y = _some_X_y()
    pline = pdp.PdPipeline([
        pdp.ApplyByCols("ph", math.ceil),
        pdp.EncodeLabel(skip=lambda X, y: 'temp' in X.columns),
    ])

    post_X, post_y = pline(X, y)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 'acd'
    assert post_y[2] == 'alk'
    assert post_y[3] == 'alk'


def test_encode_label_fit():
    X, y = _some_X_y()
    encode_stage = EncodeLabel()

    with pytest.raises(UnfittedPipelineStageError):
        encode_stage.transform(X, y)

    encode_stage.fit(X, y)
    post_X, post_y = encode_stage.transform(X, y)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 0
    assert post_y[2] == 1
    assert post_y[3] == 1

    # see only transform (no fit) when already fitted
    X2, y2 = _some_X_y2()
    post_X, post_y = encode_stage.transform(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1


def _bad_len_X_y():
    X = pd.DataFrame(
        data=[[3.2, 31], [7.2, 33], [12.1, 28]],
        index=[1, 2, 3],
        columns=["ph", "temp"]
    )
    y = pd.Series(["acd", "alk"])
    return X, y


def test_encode_label_bad_len():
    X, y = _bad_len_X_y()
    encode_stage = EncodeLabel()

    with pytest.raises(ValueError):
        encode_stage.fit(X, y)

    with pytest.raises(ValueError):
        encode_stage(X, y)


def _np_X_y():
    X = pd.DataFrame(
        data=[[3.2, 31], [7.2, 33], [12.1, 28]],
        index=[1, 2, 3],
        columns=["ph", "temp"]
    )
    y = np.array(["acd", "alk", "alk"])
    return X, y


def _np_X_y2():
    X = pd.DataFrame(
        data=[[4.1, 22], [7.7, 23], [2.1, 38]],
        index=[1, 2, 3],
        columns=["ph", "temp"]
    )
    y = np.array(["alk", "acd", "alk"])
    return X, y


def test_encode_label_np_arr():
    X, y = _np_X_y()
    encode_stage = EncodeLabel()

    post_X, post_y = encode_stage(X, y)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 0
    assert post_y[2] == 1
    assert post_y[3] == 1

    # see only transform (no fit) when already fitted
    X2, y2 = _np_X_y2()
    post_X, post_y = encode_stage(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1

    # see only transform (no fit) when already fitted
    X2, y2 = _np_X_y2()
    post_X, post_y = encode_stage.transform(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1

    # check fit_transform when already fitted
    X2, y2 = _np_X_y2()
    post_X, post_y = encode_stage.fit_transform(X2, y2)
    assert ['ph', 'temp'] == list(post_X.columns)
    assert post_y[1] == 1
    assert post_y[2] == 0
    assert post_y[3] == 1
