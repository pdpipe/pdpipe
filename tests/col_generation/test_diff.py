"""Testing Diff pipeline stages."""

import pickle

import numpy as np
import pandas as pd
import pdpipe as pdp

from pdpipe.col_generation import Diff

from pdptestutil import random_pickle_path


def diff_df():
    return pd.DataFrame(
        data=[
            [0, 10, 100, "x"],
            [1, 13, 95, "x"],
            [2, 17, 91, "y"],
            [3, 20, 90, "y"],
        ],
        index=["r0", "r1", "r2", "r3"],
        columns=["time", "sensor_a", "sensor_b", "label"],
    )


def test_diff_single_column():
    df = diff_df()
    res_df = pdp.Diff("sensor_a").apply(df)
    assert list(res_df.columns) == list(df.columns)
    assert list(res_df.index) == list(df.index)
    assert np.isnan(res_df["sensor_a"]["r0"])
    assert res_df["sensor_a"]["r1"] == 3
    assert res_df["sensor_a"]["r2"] == 4
    assert res_df["sensor_a"]["r3"] == 3
    assert res_df["label"].equals(df["label"])


def test_diff_multiple_columns():
    df = diff_df()
    res_df = Diff(["sensor_a", "sensor_b"]).apply(df)
    assert list(res_df.columns) == list(df.columns)
    assert np.isnan(res_df["sensor_a"]["r0"])
    assert res_df["sensor_a"]["r1"] == 3
    assert res_df["sensor_a"]["r2"] == 4
    assert res_df["sensor_a"]["r3"] == 3
    assert np.isnan(res_df["sensor_b"]["r0"])
    assert res_df["sensor_b"]["r1"] == -5
    assert res_df["sensor_b"]["r2"] == -4
    assert res_df["sensor_b"]["r3"] == -1


def test_diff_preserves_passthrough_columns_and_order():
    df = diff_df()
    res_df = Diff("sensor_a", drop=False).apply(df)
    assert list(res_df.columns) == [
        "time",
        "sensor_a",
        "sensor_a_diff",
        "sensor_b",
        "label",
    ]
    assert res_df["sensor_a"].equals(df["sensor_a"])
    assert np.isnan(res_df["sensor_a_diff"]["r0"])
    assert res_df["sensor_a_diff"]["r1"] == 3
    assert res_df["sensor_a_diff"]["r2"] == 4
    assert res_df["sensor_a_diff"]["r3"] == 3
    assert res_df["time"].equals(df["time"])
    assert res_df["sensor_b"].equals(df["sensor_b"])
    assert res_df["label"].equals(df["label"])


def test_diff_with_custom_suffix():
    df = diff_df()
    res_df = Diff("sensor_a", drop=False, suffix="_delta").apply(df)
    assert list(res_df.columns) == [
        "time",
        "sensor_a",
        "sensor_a_delta",
        "sensor_b",
        "label",
    ]
    assert res_df["sensor_a"].equals(df["sensor_a"])
    assert np.isnan(res_df["sensor_a_delta"]["r0"])
    assert res_df["sensor_a_delta"]["r1"] == 3
    assert res_df["sensor_a_delta"]["r2"] == 4
    assert res_df["sensor_a_delta"]["r3"] == 3


def test_diff_periods_and_result_column():
    df = diff_df()
    res_df = Diff(
        "sensor_a",
        periods=2,
        result_columns="sensor_a_delta2",
    ).apply(df)
    assert list(res_df.columns) == [
        "time",
        "sensor_a_delta2",
        "sensor_b",
        "label",
    ]
    assert np.isnan(res_df["sensor_a_delta2"]["r0"])
    assert np.isnan(res_df["sensor_a_delta2"]["r1"])
    assert res_df["sensor_a_delta2"]["r2"] == 7
    assert res_df["sensor_a_delta2"]["r3"] == 7


def test_diff_dynamic_callable_selector():
    df = diff_df()

    def selector(X):
        return [col for col in X.columns if col.startswith("sensor")]

    res_df = Diff(selector).apply(df)
    assert list(res_df.columns) == list(df.columns)
    assert np.isnan(res_df["sensor_a"]["r0"])
    assert res_df["sensor_a"]["r1"] == 3
    assert np.isnan(res_df["sensor_b"]["r0"])
    assert res_df["sensor_b"]["r1"] == -5
    assert res_df["time"].equals(df["time"])
    assert res_df["label"].equals(df["label"])


def test_diff_column_qualifier_selector():
    df = diff_df()
    res_df = Diff(pdp.cq.ByLabels(["sensor_a", "sensor_b"])).apply(df)
    assert list(res_df.columns) == list(df.columns)
    assert np.isnan(res_df["sensor_a"]["r0"])
    assert res_df["sensor_a"]["r1"] == 3
    assert np.isnan(res_df["sensor_b"]["r0"])
    assert res_df["sensor_b"]["r1"] == -5


def test_pickle_diff(pdpipe_tests_dir_path):
    df = diff_df()
    stage = Diff("sensor_a", periods=2, drop=False, suffix="_delta2")
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage(df)
    assert list(res_df.columns) == [
        "time",
        "sensor_a",
        "sensor_a_delta2",
        "sensor_b",
        "label",
    ]
    assert np.isnan(res_df["sensor_a_delta2"]["r0"])
    assert np.isnan(res_df["sensor_a_delta2"]["r1"])
    assert res_df["sensor_a_delta2"]["r2"] == 7
    assert res_df["sensor_a_delta2"]["r3"] == 7
