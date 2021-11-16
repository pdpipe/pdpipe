"""Testing MapColVals pipeline stages."""

import datetime

import pandas as pd
import pytest

from pdpipe.col_generation import MapColVals


def _test_df():
    return pd.DataFrame([[1], [3], [2]], ['UK', 'USSR', 'US'], ['Medal'])


def test_mapcolvals():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    res_df = MapColVals('Medal', value_map).apply(df)
    assert res_df['Medal']['UK'] == 'Gold'
    assert res_df['Medal']['USSR'] == 'Bronze'
    assert res_df['Medal']['US'] == 'Silver'


def test_mapcolvals_no_drop():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    res_df = MapColVals('Medal', value_map, drop=False).apply(df)
    assert res_df['Medal']['UK'] == 1
    assert res_df['Medal']['USSR'] == 3
    assert res_df['Medal']['US'] == 2
    assert res_df['Medal_map']['UK'] == 'Gold'
    assert res_df['Medal_map']['USSR'] == 'Bronze'
    assert res_df['Medal_map']['US'] == 'Silver'


def test_mapcolvals_with_res_name():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    res_df = MapColVals('Medal', value_map, result_columns='Metal').apply(df)
    assert res_df['Metal']['UK'] == 'Gold'
    assert res_df['Metal']['USSR'] == 'Bronze'
    assert res_df['Metal']['US'] == 'Silver'


def test_mapcolvals_with_res_name_no_drop():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    map_stage = MapColVals(
        'Medal', value_map, result_columns='Metal', drop=False)
    res_df = map_stage(df)
    assert res_df['Medal']['UK'] == 1
    assert res_df['Medal']['USSR'] == 3
    assert res_df['Medal']['US'] == 2
    assert res_df['Metal']['UK'] == 'Gold'
    assert res_df['Metal']['USSR'] == 'Bronze'
    assert res_df['Metal']['US'] == 'Silver'


def test_mapcolvals_bad_res_name_len():
    """Testing MapColVals pipeline stages."""
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    with pytest.raises(ValueError):
        map_stage = MapColVals('Medal', value_map, result_columns=['A', 'B'])
        assert isinstance(map_stage, MapColVals)


def _2nd_test_df():
    return pd.DataFrame(
        data=[
            [datetime.timedelta(days=2, minutes=5)],
            [datetime.timedelta(days=3, minutes=10)],
            [datetime.timedelta(days=4, minutes=15)]
        ],
        index=['UK', 'USSR', 'US'],
        columns=['Duration'],
    )


def test_mapcolvals_with_attr_name():
    """Testing MapColVals pipeline stages."""
    df = _2nd_test_df()
    res_df = MapColVals('Duration', 'days').apply(df)
    assert res_df['Duration']['UK'] == 2
    assert res_df['Duration']['USSR'] == 3
    assert res_df['Duration']['US'] == 4


def test_mapcolvals_with_method_name():
    """Testing MapColVals pipeline stages."""
    df = _2nd_test_df()
    res_df = MapColVals('Duration', ('total_seconds', {})).apply(df)
    assert res_df['Duration']['UK'] == df['Duration']['UK'].total_seconds()
    assert res_df['Duration']['USSR'] == df['Duration']['USSR'].total_seconds()
    assert res_df['Duration']['US'] == df['Duration']['US'].total_seconds()


def _3rd_test_df():
    return pd.DataFrame(
        data=[
            [datetime.timedelta(weeks=2)],
            [datetime.timedelta(weeks=4)],
            [datetime.timedelta(weeks=10)]
        ],
        index=['proposal', 'midterm', 'finals'],
        columns=['Due'],
    )


def test_mapcolvals_with_method_name_for_documentation():
    """Testing MapColVals pipeline stages."""
    df = _3rd_test_df()
    res_df = MapColVals('Due', ('total_seconds', {})).apply(df)
    assert res_df['Due']['proposal'] == df['Due']['proposal'].total_seconds()
    assert res_df['Due']['midterm'] == df['Due']['midterm'].total_seconds()
    assert res_df['Due']['finals'] == df['Due']['finals'].total_seconds()
