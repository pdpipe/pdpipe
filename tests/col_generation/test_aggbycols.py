"""Testing AggByCols pipeline stages."""

import pytest
import numpy as np
from numpy.testing import assert_approx_equal
import pandas as pd

from pdpipe import AggByCols


_LOG32 = 1.163151
_LOG72 = 1.974081
_LOG121 = 2.493205


def ph_df():
    return pd.DataFrame(
        data=[[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]],
        index=[1, 2, 3],
        columns=["ph", "lbl"],
    )


def test_aggbycols_func_series():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", "min")
    res_df = min_ph(df)
    assert res_df.columns.get_loc("ph") == 0
    assert res_df["ph"][1] == 3.2
    assert res_df["ph"][2] == 3.2
    assert res_df["ph"][3] == 3.2


def test_aggbycols_func_scaler():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", "min")
    res_df = min_ph(df)
    assert res_df.columns.get_loc("ph") == 0
    assert res_df["ph"][1] == 3.2
    assert res_df["ph"][2] == 3.2
    assert res_df["ph"][3] == 3.2


def test_aggbycols_func_desc():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", "min", func_desc="Minimum PH value")
    res_df = min_ph(df)
    assert res_df.columns.get_loc("ph") == 0
    assert res_df["ph"][1] == 3.2
    assert res_df["ph"][2] == 3.2
    assert res_df["ph"][3] == 3.2


def test_aggbycols_with_result_columns():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", "min", result_columns="min_ph")
    res_df = min_ph(df)
    assert "ph" not in res_df.columns
    assert res_df.columns.get_loc("min_ph") == 0
    assert res_df["min_ph"][1] == 3.2
    assert res_df["min_ph"][2] == 3.2
    assert res_df["min_ph"][3] == 3.2


def test_aggbycols_with_drop():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", "min", drop=False)
    res_df = min_ph(df)
    assert "ph" in res_df.columns
    assert "ph_agg" in res_df.columns
    assert res_df.columns.get_loc("ph") == 0
    assert res_df.columns.get_loc("ph_agg") == 1
    assert res_df["ph_agg"][1] == 3.2
    assert res_df["ph_agg"][2] == 3.2
    assert res_df["ph_agg"][3] == 3.2


def test_aggbycols_no_drop_custom_suffix():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", "min", drop=False, suffix="_min")
    res_df = min_ph(df)
    assert "ph" in res_df.columns
    assert "ph_min" in res_df.columns
    assert res_df.columns.get_loc("ph") == 0
    assert res_df.columns.get_loc("ph_min") == 1
    assert res_df["ph_min"][1] == 3.2
    assert res_df["ph_min"][2] == 3.2
    assert res_df["ph_min"][3] == 3.2


def test_aggbycols_with_bad_len_result_columns():
    """Testing ApplyByCols pipeline stages."""
    with pytest.raises(ValueError):
        AggByCols("ph", "min", result_columns=["a", "b"])
