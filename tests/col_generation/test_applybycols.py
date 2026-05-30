"""Testing ApplyByCols pipeline stages."""

import math
import pickle

import pytest
import numpy as np
import pandas as pd
import pdpipe as pdp

from pdpipe.col_generation import ApplyByCols

from pdptestutil import random_pickle_path


def ph_df():
    return pd.DataFrame(
        data=[[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]],
        index=[1, 2, 3],
        columns=["ph", "lbl"],
    )


def multi_col_df():
    return pd.DataFrame(
        data=[
            ["x", 1, "m1", 10, "z1"],
            ["y", 2, "m2", 20, "z2"],
            ["z", 3, "m3", 30, "z3"],
        ],
        index=["row-c", "row-a", "row-b"],
        columns=["id", "a", "middle", "b", "tail"],
    )


def _add_by_label(value, offset=0, label=None):
    label_offset = {"a": 100, "b": 200}[label]
    return value + offset + label_offset


def _subtract_context_offset(value, label, application_context):
    return value - application_context["offsets"][label]


def test_applybycols():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil)
    res_df = round_ph(df)
    assert res_df.columns.get_loc("ph") == 0
    assert res_df["ph"][1] == 4
    assert res_df["ph"][2] == 8
    assert res_df["ph"][3] == 13


def test_applybycols_func_desc():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil, func_desc="Round PH values")
    res_df = round_ph(df)
    assert res_df.columns.get_loc("ph") == 0
    assert res_df["ph"][1] == 4
    assert res_df["ph"][2] == 8
    assert res_df["ph"][3] == 13


def test_applybycols_with_result_columns():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil, result_columns="round_ph")
    res_df = round_ph(df)
    assert "ph" not in res_df.columns
    assert res_df.columns.get_loc("round_ph") == 0
    assert res_df["round_ph"][1] == 4
    assert res_df["round_ph"][2] == 8
    assert res_df["round_ph"][3] == 13


def test_applybycols_with_drop():
    """Testing ApplyByCols pipeline stages."""
    df = ph_df()
    round_ph = ApplyByCols("ph", math.ceil, drop=False)
    res_df = round_ph(df)
    assert "ph" in res_df.columns
    assert "ph_app" in res_df.columns
    assert res_df.columns.get_loc("ph") == 0
    assert res_df.columns.get_loc("ph_app") == 1
    assert res_df["ph_app"][1] == 4
    assert res_df["ph_app"][2] == 8
    assert res_df["ph_app"][3] == 13


def test_applybycols_with_bad_len_result_columns():
    """Testing ApplyByCols pipeline stages."""
    with pytest.raises(ValueError):
        ApplyByCols("ph", math.ceil, result_columns=["a", "b"])


def test_applybycols_parallel_matches_serial():
    """Testing opt-in ApplyByCols parallel execution."""
    df = multi_col_df()
    serial_stage = ApplyByCols(
        ["a", "b"],
        _add_by_label,
        drop=False,
        suffix="_app",
        args=(5,),
        n_jobs=1,
    )
    parallel_stage = ApplyByCols(
        ["a", "b"],
        _add_by_label,
        drop=False,
        suffix="_app",
        args=(5,),
        n_jobs=2,
    )
    serial_df = serial_stage(df)
    parallel_df = parallel_stage(df)
    pd.testing.assert_frame_equal(parallel_df, serial_df)


def test_applybycols_parallel_with_result_columns_matches_serial():
    """Testing ApplyByCols parallel custom result column names."""
    df = multi_col_df()
    serial_stage = ApplyByCols(
        ["a", "b"],
        _add_by_label,
        result_columns=["a_new", "b_new"],
        args=(5,),
        n_jobs=1,
    )
    parallel_stage = ApplyByCols(
        ["a", "b"],
        _add_by_label,
        result_columns=["a_new", "b_new"],
        args=(5,),
        n_jobs=2,
    )
    serial_df = serial_stage(df)
    parallel_df = parallel_stage(df)
    pd.testing.assert_frame_equal(parallel_df, serial_df)
    assert list(parallel_df.columns) == [
        "id",
        "a_new",
        "middle",
        "b_new",
        "tail",
    ]


def test_applybycols_parallel_n_jobs_minus_one_matches_serial():
    """Testing ApplyByCols n_jobs=-1 compatibility."""
    df = multi_col_df()
    serial_stage = ApplyByCols(["a", "b"], _add_by_label, n_jobs=1)
    parallel_stage = ApplyByCols(["a", "b"], _add_by_label, n_jobs=-1)
    pd.testing.assert_frame_equal(parallel_stage(df), serial_stage(df))


def test_applybycols_parallel_rejects_invalid_n_jobs():
    """Testing ApplyByCols n_jobs validation."""
    df = multi_col_df()
    with pytest.raises(TypeError):
        ApplyByCols(["a", "b"], _add_by_label, n_jobs="2")(df)
    with pytest.raises(ValueError):
        ApplyByCols(["a", "b"], _add_by_label, n_jobs=0)(df)


def test_applybycols_parallel_preserves_column_order_and_index():
    """Testing ApplyByCols parallel output shape invariants."""
    df = multi_col_df()
    parallel_stage = ApplyByCols(
        ["a", "b"],
        _add_by_label,
        drop=False,
        suffix="_parallel",
        args=(5,),
        n_jobs=2,
    )
    res_df = parallel_stage(df)
    assert list(res_df.index) == ["row-c", "row-a", "row-b"]
    assert list(res_df.columns) == [
        "id",
        "a",
        "a_parallel",
        "middle",
        "b_parallel",
        "b",
        "tail",
    ]


def test_applybycols_n_jobs_none_and_one_use_serial_behavior():
    """Testing ApplyByCols default n_jobs compatibility."""
    df = multi_col_df()
    default_stage = ApplyByCols(["a", "b"], _add_by_label)
    none_stage = ApplyByCols(["a", "b"], _add_by_label, n_jobs=None)
    one_stage = ApplyByCols(["a", "b"], _add_by_label, n_jobs=1)
    default_df = default_stage(df)
    pd.testing.assert_frame_equal(none_stage(df), default_df)
    pd.testing.assert_frame_equal(one_stage(df), default_df)


def test_applybycols_parallel_matches_serial_with_application_context():
    """Testing ApplyByCols parallel context injection."""
    df = multi_col_df()
    serial_pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(
                offsets=lambda X: {col: X[col].min() for col in ["a", "b"]},
            ),
            pdp.ApplyByCols(
                ["a", "b"],
                _subtract_context_offset,
                n_jobs=1,
            ),
        ]
    )
    parallel_pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(
                offsets=lambda X: {col: X[col].min() for col in ["a", "b"]},
            ),
            pdp.ApplyByCols(
                ["a", "b"],
                _subtract_context_offset,
                n_jobs=2,
            ),
        ]
    )
    pd.testing.assert_frame_equal(parallel_pipeline(df), serial_pipeline(df))


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


DF1 = pd.DataFrame({"a": ["a", "b", "c", "d"], "b": [5, 6, 7, 1]})


def test_complex_drop():
    pline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(
                numeric_means=lambda df: df.select_dtypes(include=np.number)
                .mean()
                .to_dict(),
            ),
            pdp.ApplyByCols(
                columns=pdp.cq.OfNumericDtypes(),
                func=lambda x, label, application_context: (
                    "DROP"
                    if x < application_context["numeric_means"][label]
                    else x
                ),
            ),
            pdp.ValDrop(["DROP"]),
        ]
    )
    res = pline(DF1)
    assert res.index.tolist() == [0, 1, 2]
    assert 1 not in res["b"].values


def test_applybycols_use_fit_context():
    pline = pdp.PdPipeline(
        [
            pdp.ApplyByCols(
                columns=pdp.cq.OfNumericDtypes(),
                func=lambda x, label, fit_context: (
                    "BLAH" if (label in fit_context.keys()) else x
                ),
            ),
        ]
    )
    res = pline(DF1)
    assert res.index.tolist() == [0, 1, 2, 3]
    assert "BLAH" not in res["a"].values
    assert "BLAH" not in res["b"].values


def test_pickle_applybycols(pdpipe_tests_dir_path):
    """Testing ApplyByCols pickling."""
    df = ph_df()
    stage = ApplyByCols("ph", math.ceil)
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage(df)
    assert res_df["ph"][1] == 4
    assert res_df["ph"][2] == 8
    assert res_df["ph"][3] == 13
