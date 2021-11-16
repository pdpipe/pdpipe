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
        columns=["ph", "lbl"]
    )


def test_aggbycols_func_series():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    log_ph = AggByCols("ph", np.log)
    res_df = log_ph(df)
    assert res_df.columns.get_loc('ph') == 0
    assert_approx_equal(res_df['ph'][1], _LOG32, significant=5)
    assert_approx_equal(res_df['ph'][2], _LOG72, significant=5)
    assert_approx_equal(res_df['ph'][3], _LOG121, significant=5)


def test_aggbycols_func_scaler():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    min_ph = AggByCols("ph", min)
    res_df = min_ph(df)
    assert res_df.columns.get_loc('ph') == 0
    assert res_df['ph'][1], min(res_df['ph'])
    assert res_df['ph'][2], min(res_df['ph'])
    assert res_df['ph'][3], min(res_df['ph'])


def test_aggbycols_func_desc():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    log_ph = AggByCols("ph", np.log, func_desc='Round PH values')
    res_df = log_ph(df)
    assert res_df.columns.get_loc('ph') == 0
    assert_approx_equal(res_df['ph'][1], _LOG32, significant=5)
    assert_approx_equal(res_df['ph'][2], _LOG72, significant=5)
    assert_approx_equal(res_df['ph'][3], _LOG121, significant=5)


def test_aggbycols_with_result_columns():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    log_ph = AggByCols("ph", np.log, result_columns='log_ph')
    res_df = log_ph(df)
    assert 'ph' not in res_df.columns
    assert res_df.columns.get_loc('log_ph') == 0
    assert_approx_equal(res_df['log_ph'][1], _LOG32, significant=5)
    assert_approx_equal(res_df['log_ph'][2], _LOG72, significant=5)
    assert_approx_equal(res_df['log_ph'][3], _LOG121, significant=5)


def test_aggbycols_with_drop():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    log_ph = AggByCols("ph", np.log, drop=False)
    res_df = log_ph(df)
    assert 'ph' in res_df.columns
    assert 'ph_agg' in res_df.columns
    assert res_df.columns.get_loc('ph') == 0
    assert res_df.columns.get_loc('ph_agg') == 1
    assert_approx_equal(res_df['ph_agg'][1], _LOG32, significant=5)
    assert_approx_equal(res_df['ph_agg'][2], _LOG72, significant=5)
    assert_approx_equal(res_df['ph_agg'][3], _LOG121, significant=5)


def test_aggbycols_no_drop_custom_suffix():
    """Testing AggByCols pipeline stages."""
    df = ph_df()
    log_ph = AggByCols("ph", np.log, drop=False, suffix='_log')
    res_df = log_ph(df)
    assert 'ph' in res_df.columns
    assert 'ph_log' in res_df.columns
    assert res_df.columns.get_loc('ph') == 0
    assert res_df.columns.get_loc('ph_log') == 1
    assert_approx_equal(res_df['ph_log'][1], _LOG32, significant=5)
    assert_approx_equal(res_df['ph_log'][2], _LOG72, significant=5)
    assert_approx_equal(res_df['ph_log'][3], _LOG121, significant=5)


def test_aggbycols_with_bad_len_result_columns():
    """Testing ApplyByCols pipeline stages."""
    with pytest.raises(ValueError):
        AggByCols("ph", np.log, result_columns=['a', 'b'])
