"""Testing basic pipeline stages."""

import math

import pandas as pd

from pdpipe.col_generation import Bin


def test_col_binner():
    """Testing the col binner helper class."""
    binner = Bin._get_col_binner([0, 5])
    assert binner(-math.inf) == '<0'
    assert binner(-4) == '<0'
    assert binner(0) == '0-5'
    assert binner(1) == '0-5'
    assert binner(4.99) == '0-5'
    assert binner(5) == '5≤'
    assert binner(232) == '5≤'
    assert binner(math.inf) == '5≤'


def test_bin_verbose():
    """Basic binning test."""
    df = pd.DataFrame([[-3], [4], [5], [9]], [1, 2, 3, 4], ['speed'])
    bin_stage = Bin({'speed': [5]}, drop=False)
    res_df = bin_stage.apply(df, verbose=True)
    assert 'speed_bin' in res_df.columns
    assert res_df['speed_bin'][1] == '<5'
    assert res_df['speed_bin'][2] == '<5'
    assert res_df['speed_bin'][3] == '5≤'
    assert res_df['speed_bin'][4] == '5≤'


def test_bin_drop():
    """Basic binning test."""
    df = pd.DataFrame([[-3], [4], [5], [9]], [1, 2, 3, 4], ['speed'])
    bin_stage = Bin({'speed': [5]}, drop=True)
    res_df = bin_stage.apply(df, verbose=True)
    assert 'speed_bin' not in res_df.columns
    assert res_df['speed'][1] == '<5'
    assert res_df['speed'][2] == '<5'
    assert res_df['speed'][3] == '5≤'
    assert res_df['speed'][4] == '5≤'


def test_bin_two_col():
    """Basic binning test."""
    df = pd.DataFrame([[-3, 9], [4, 2], [5, 1], [9, 5]], columns=['s', 'p'])
    bin_stage = Bin({'s': [5], 'p': [5]}, drop=False)
    res_df = bin_stage.apply(df, verbose=True)
    assert 's_bin' in res_df.columns
    assert res_df['s_bin'][0] == '<5'
    assert res_df['s_bin'][1] == '<5'
    assert res_df['s_bin'][2] == '5≤'
    assert res_df['s_bin'][3] == '5≤'
    assert 'p_bin' in res_df.columns
    assert res_df['p_bin'][0] == '5≤'
    assert res_df['p_bin'][1] == '<5'
    assert res_df['p_bin'][2] == '<5'
    assert res_df['p_bin'][3] == '5≤'
