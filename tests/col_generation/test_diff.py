"""Tests for the Diff column-generation pipeline stage.

Covers issue #9 (Diff transformer): wrap pandas.DataFrame.diff(periods) as a
pdpipe stage so it composes with the rest of a pipeline.

"""

import pickle

import numpy as np
import pandas as pd
import pytest

import pdpipe as pdp

from pdptestutil import random_pickle_path


def _ts_df():
    # Small, reproducible time-series-shaped frame: a monotonic time
    # column (t) and a noisy value column (val) we'll detrend.
    return pd.DataFrame(
        data=[[3, 100], [5, 110], [10, 95], [12, 130]],
        index=[1, 2, 3, 4],
        columns=["t", "val"],
    )


def _ts_df2():
    return pd.DataFrame(
        data=[[2, 50], [4, 75], [9, 60]],
        index=[1, 2, 3],
        columns=["t", "val"],
    )


def test_diff_default():
    df = _ts_df()
    stage = pdp.Diff("val")
    res = stage(df)
    # Source column preserved (drop=False default), new column appended.
    assert "val" in res.columns
    assert "val_diff" in res.columns
    # First row is NaN by definition of pandas.Series.diff(periods=1).
    assert np.isnan(res["val_diff"].iloc[0])
    assert res["val_diff"].iloc[1] == 10
    assert res["val_diff"].iloc[2] == -15
    assert res["val_diff"].iloc[3] == 35
    # `t` column is untouched.
    assert (res["t"] == df["t"]).all()


def test_diff_drop_replaces_source():
    df = _ts_df()
    stage = pdp.Diff("val", drop=True)
    res = stage(df)
    # When drop=True, the source column is replaced in place — same name,
    # no _diff suffix.
    assert "val" in res.columns
    assert "val_diff" not in res.columns
    assert np.isnan(res["val"].iloc[0])
    assert res["val"].iloc[1] == 10


def test_diff_periods_negative():
    df = _ts_df()
    stage = pdp.Diff("val", periods=-1)
    res = stage(df)
    # periods=-1 means "current minus next"; last row becomes NaN.
    assert res["val_diff"].iloc[0] == -10
    assert res["val_diff"].iloc[1] == 15
    assert res["val_diff"].iloc[2] == -35
    assert np.isnan(res["val_diff"].iloc[3])


def test_diff_periods_two():
    df = _ts_df()
    stage = pdp.Diff("val", periods=2)
    res = stage(df)
    # First two rows NaN with periods=2.
    assert np.isnan(res["val_diff"].iloc[0])
    assert np.isnan(res["val_diff"].iloc[1])
    assert res["val_diff"].iloc[2] == -5  # 95 - 100
    assert res["val_diff"].iloc[3] == 20  # 130 - 110


def test_diff_default_columns_picks_numeric():
    # Mixed-dtype frame: when columns is None, only numeric columns are
    # differenced (matches Log/AggByCols convention).
    df = pd.DataFrame(
        data=[[1, 2.0, "a"], [4, 5.0, "b"], [9, 8.0, "c"]],
        columns=["i", "f", "s"],
    )
    stage = pdp.Diff()
    res = stage(df)
    assert "i_diff" in res.columns
    assert "f_diff" in res.columns
    # String column must not be differenced.
    assert "s_diff" not in res.columns


def test_diff_custom_suffix():
    df = _ts_df()
    stage = pdp.Diff("val", suffix="_delta")
    res = stage(df)
    assert "val_delta" in res.columns
    assert "val_diff" not in res.columns


def test_diff_invalid_periods_raises():
    # Validate at construction so failures don't hide until apply time.
    with pytest.raises(TypeError):
        pdp.Diff("val", periods=1.5)
    with pytest.raises(TypeError):
        pdp.Diff("val", periods="1")


def test_diff_transform_after_fit():
    # Diff is stateless, so transform on a fresh frame must produce the
    # same shape as fit_transform (no leftover fit-time state leaks).
    stage = pdp.Diff("val")
    stage.fit_transform(_ts_df())
    res2 = stage.transform(_ts_df2())
    assert "val_diff" in res2.columns
    assert np.isnan(res2["val_diff"].iloc[0])
    assert res2["val_diff"].iloc[1] == 25
    assert res2["val_diff"].iloc[2] == -15


def test_diff_in_pipeline():
    # Compose with other stages — the load-bearing requirement of issue #9
    # ("Scikit like transformer ... include into a clean pipeline").
    df = _ts_df()
    pipeline = pdp.PdPipeline(
        stages=[
            pdp.Diff("val"),
            pdp.ColDrop("val"),
        ]
    )
    res = pipeline(df)
    assert "val" not in res.columns
    assert "val_diff" in res.columns


def test_diff_pickle(pdpipe_tests_dir_path):
    """Diff must round-trip through pickle, like every other stage."""
    df = _ts_df()
    stage = pdp.Diff("val")
    stage(df)
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded = pickle.load(f)
    res2 = loaded(_ts_df2())
    assert "val_diff" in res2.columns
