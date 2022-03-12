"""Testing ApplyByCols pipeline stages."""

import math

import pytest
import numpy as np
import pandas as pd
import pdpipe as pdp

from pdpipe.col_generation import ApplyByCols


def ph_df():
    return pd.DataFrame(
        data=[[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]],
        index=[1, 2, 3],
        columns=["ph", "lbl"]
    )


def test_applybycols():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil)
    res_df = round_ph(df)
    assert res_df.columns.get_loc('ph') == 0
    assert res_df['ph'][1] == 4
    assert res_df['ph'][2] == 8
    assert res_df['ph'][3] == 13


def test_applybycols_func_desc():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil, func_desc='Round PH values')
    res_df = round_ph(df)
    assert res_df.columns.get_loc('ph') == 0
    assert res_df['ph'][1] == 4
    assert res_df['ph'][2] == 8
    assert res_df['ph'][3] == 13


def test_applybycols_with_result_columns():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil, result_columns='round_ph')
    res_df = round_ph(df)
    assert 'ph' not in res_df.columns
    assert res_df.columns.get_loc('round_ph') == 0
    assert res_df['round_ph'][1] == 4
    assert res_df['round_ph'][2] == 8
    assert res_df['round_ph'][3] == 13


def test_applybycols_with_drop():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil, drop=False)
    res_df = round_ph(df)
    assert 'ph' in res_df.columns
    assert 'ph_app' in res_df.columns
    assert res_df.columns.get_loc('ph') == 0
    assert res_df.columns.get_loc('ph_app') == 1
    assert res_df['ph_app'][1] == 4
    assert res_df['ph_app'][2] == 8
    assert res_df['ph_app'][3] == 13


def test_applybycols_with_bad_len_result_columns():
    """Testing ApplyByCols pipeline stages."""
    with pytest.raises(ValueError):
        ApplyByCols("ph", math.ceil, result_columns=['a', 'b'])


# def _num_df():
#     return pd.DataFrame(
#         data=[[1, 2, 'a'], [2, 4, 'b']],
#         index=[1, 2],
#         columns=['num1', 'num2', 'char']
#     )


# def _sum_and_diff(row):
#     return pd.Series({
#         'sum': row['num1'] + row['num2'],
#         'diff': row['num1'] - row['num2']
#     })


# def test_applytorows_with_df_generation():
#     """Testing ApplyToRows pipeline stages."""
#     df = _num_df()
#     cbf_stage = ApplyToRows(_sum_and_diff)
#     res_df = cbf_stage(df)
#     assert 'sum' in res_df.columns
#     assert 'diff' in res_df.columns
#     assert res_df['sum'][1] == 3
#     assert res_df['sum'][2] == 6
#     assert res_df['diff'][1] == -1
#     assert res_df['diff'][2] == -2


# def test_applytorows_with_df_generation_and_optionals():
#     """Testing ApplyToRows pipeline stages."""
#     df = _num_df()
#     cbf_stage = ApplyToRows(
#         func=_sum_and_diff,
#         func_desc="calculates sum and diff of num1 and num2",
#         prec=lambda df: 'num1' in df.columns and 'num2' in df.columns
#     )
#     res_df = cbf_stage(df)
#     assert 'sum' in res_df.columns
#     assert 'diff' in res_df.columns
#     assert res_df['sum'][1] == 3
#     assert res_df['sum'][2] == 6
#     assert res_df['diff'][1] == -1
#     assert res_df['diff'][2] == -2


# def test_applytorows_with_df_generation_follow():
#     """Testing ApplyToRows pipeline stages."""
#     df = _num_df()
#     cbf_stage = ApplyToRows(_sum_and_diff, follow_column='num1')
#     res_df = cbf_stage(df)
#     assert res_df.columns.get_loc('sum') == 2
#     assert res_df.columns.get_loc('diff') == 1
#     assert 'sum' in res_df.columns
#     assert 'diff' in res_df.columns
#     assert res_df['sum'][1] == 3
#     assert res_df['sum'][2] == 6
#     assert res_df['diff'][1] == -1
#     assert res_df['diff'][2] == -2


DF1 = pd.DataFrame({'a': ['a', 'b', 'c', 'd'], 'b': [5, 6, 7, 1]})


def test_complex_drop():
    pline = pdp.PdPipeline([
        pdp.ApplicationContextEnricher(
            numeric_means=lambda df:
                df.select_dtypes(include=np.number).mean().to_dict(),
        ),
        pdp.ApplyByCols(
            columns=pdp.cq.OfNumericDtypes(),
            func=lambda x, label, application_context:
                'DROP' if x < application_context['numeric_means'][label]
                else x,
        ),
        pdp.ValDrop(['DROP']),
    ])
    res = pline(DF1)
    assert res.index.tolist() == [0, 1, 2]
    assert 1 not in res['b'].values


def test_applybycols_use_fit_context():
    pline = pdp.PdPipeline([
        pdp.ApplyByCols(
            columns=pdp.cq.OfNumericDtypes(),
            func=lambda x, label, fit_context:
                'BLAH' if (label in fit_context.keys()) else x,
        ),
    ])
    res = pline(DF1)
    assert res.index.tolist() == [0, 1, 2, 3]
    assert 'BLAH' not in res['a'].values
    assert 'BLAH' not in res['b'].values
