"""Testing ApplyToRows pipeline stages."""

import pandas as pd

from pdpipe.col_generation import ApplyToRows


def _some_df():
    return pd.DataFrame(
        data=[[3, 2143], [10, 1321], [7, 1255]],
        index=[1, 2, 3],
        columns=["years", "avg_revenue"]
    )


def _total_rev(row):
    return row['years'] * row['avg_revenue']


def test_applytorows():
    """Testing ApplyToRows pipeline stages."""
    df = _some_df()
    cbf_stage = ApplyToRows(_total_rev, 'total_revenue')
    res_df = cbf_stage(df)
    assert res_df.columns.get_loc('total_revenue') == 2
    assert res_df['total_revenue'][1] == 3 * 2143
    assert res_df['total_revenue'][2] == 10 * 1321
    assert res_df['total_revenue'][3] == 7 * 1255


def test_applytorows_with_follow_column():
    """Testing ApplyToRows pipeline stages."""
    df = _some_df()
    cbf_stage = ApplyToRows(_total_rev, 'total_revenue', follow_column='years')
    res_df = cbf_stage(df)
    assert res_df.columns.get_loc('total_revenue') == 1
    assert res_df['total_revenue'][1] == 3 * 2143
    assert res_df['total_revenue'][2] == 10 * 1321
    assert res_df['total_revenue'][3] == 7 * 1255


def _num_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


def _sum_and_diff(row):
    return pd.Series(
        {'sum': row['num1'] + row['num2'], 'diff': row['num1'] - row['num2']})


def test_applytorows_with_df_generation():
    """Testing ApplyToRows pipeline stages."""
    df = _num_df()
    cbf_stage = ApplyToRows(_sum_and_diff)
    res_df = cbf_stage(df)
    assert 'sum' in res_df.columns
    assert 'diff' in res_df.columns
    assert res_df['sum'][1] == 3
    assert res_df['sum'][2] == 6
    assert res_df['diff'][1] == -1
    assert res_df['diff'][2] == -2


def test_applytorows_with_df_generation_and_optionals():
    """Testing ApplyToRows pipeline stages."""
    df = _num_df()
    cbf_stage = ApplyToRows(
        func=_sum_and_diff,
        func_desc="calculates sum and diff of num1 and num2",
        prec=lambda df: 'num1' in df.columns and 'num2' in df.columns
    )
    res_df = cbf_stage(df)
    assert 'sum' in res_df.columns
    assert 'diff' in res_df.columns
    assert res_df['sum'][1] == 3
    assert res_df['sum'][2] == 6
    assert res_df['diff'][1] == -1
    assert res_df['diff'][2] == -2


def test_applytorows_with_df_generation_follow():
    """Testing ApplyToRows pipeline stages."""
    df = _num_df()
    cbf_stage = ApplyToRows(_sum_and_diff, follow_column='num1')
    res_df = cbf_stage(df)
    print(res_df.columns)
    assert res_df.columns.get_loc('sum') == 2
    assert res_df.columns.get_loc('diff') == 1
    assert 'sum' in res_df.columns
    assert 'diff' in res_df.columns
    assert res_df['sum'][1] == 3
    assert res_df['sum'][2] == 6
    assert res_df['diff'][1] == -1
    assert res_df['diff'][2] == -2
