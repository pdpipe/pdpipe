"""Testing ApplyToRows pipeline stages."""

import pickle

import pandas as pd

from pdpipe.col_generation import ApplyToRows

from pdptestutil import random_pickle_path


def _some_df():
    return pd.DataFrame(
        data=[[3, 2143], [10, 1321], [7, 1255]],
        index=[1, 2, 3],
        columns=["years", "avg_revenue"],
    )


def _total_rev(row):
    return row["years"] * row["avg_revenue"]


def test_applytorows():
    """Testing ApplyToRows pipeline stages."""
    df = _some_df()
    cbf_stage = ApplyToRows(_total_rev, "total_revenue")
    res_df = cbf_stage(df)
    assert res_df.columns.get_loc("total_revenue") == 2
    assert res_df["total_revenue"][1] == 3 * 2143
    assert res_df["total_revenue"][2] == 10 * 1321
    assert res_df["total_revenue"][3] == 7 * 1255


def test_applytorows_with_follow_column():
    """Testing ApplyToRows pipeline stages."""
    df = _some_df()
    cbf_stage = ApplyToRows(_total_rev, "total_revenue", follow_column="years")
    res_df = cbf_stage(df)
    assert res_df.columns.get_loc("total_revenue") == 1
    assert res_df["total_revenue"][1] == 3 * 2143
    assert res_df["total_revenue"][2] == 10 * 1321
    assert res_df["total_revenue"][3] == 7 * 1255


def _num_df():
    return pd.DataFrame(
        data=[[1, 2, "a"], [2, 4, "b"]],
        index=[1, 2],
        columns=["num1", "num2", "char"],
    )


def _sum_and_diff(row):
    return pd.Series(
        {"sum": row["num1"] + row["num2"], "diff": row["num1"] - row["num2"]}
    )


def test_applytorows_with_df_generation():
    """Testing ApplyToRows pipeline stages."""
    df = _num_df()
    cbf_stage = ApplyToRows(_sum_and_diff)
    res_df = cbf_stage(df)
    assert "sum" in res_df.columns
    assert "diff" in res_df.columns
    assert res_df["sum"][1] == 3
    assert res_df["sum"][2] == 6
    assert res_df["diff"][1] == -1
    assert res_df["diff"][2] == -2


def test_applytorows_with_df_generation_and_optionals():
    """Testing ApplyToRows pipeline stages."""
    df = _num_df()
    cbf_stage = ApplyToRows(
        func=_sum_and_diff,
        func_desc="calculates sum and diff of num1 and num2",
        prec=lambda df: "num1" in df.columns and "num2" in df.columns,
    )
    res_df = cbf_stage(df)
    assert "sum" in res_df.columns
    assert "diff" in res_df.columns
    assert res_df["sum"][1] == 3
    assert res_df["sum"][2] == 6
    assert res_df["diff"][1] == -1
    assert res_df["diff"][2] == -2


def test_applytorows_with_df_generation_follow():
    """Testing ApplyToRows pipeline stages."""
    df = _num_df()
    cbf_stage = ApplyToRows(_sum_and_diff, follow_column="num1")
    res_df = cbf_stage(df)
    print(res_df.columns)
    assert res_df.columns.get_loc("sum") == 2
    assert res_df.columns.get_loc("diff") == 1
    assert "sum" in res_df.columns
    assert "diff" in res_df.columns
    assert res_df["sum"][1] == 3
    assert res_df["sum"][2] == 6
    assert res_df["diff"][1] == -1
    assert res_df["diff"][2] == -2


def test_pickle_applytorows(pdpipe_tests_dir_path):
    """Testing ApplyToRows pickling."""
    df = _some_df()
    stage = ApplyToRows(_total_rev, "total_revenue")
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    res_df = loaded_stage(df)
    assert "total_revenue" in res_df.columns
    assert res_df["total_revenue"][1] == 3 * 2143
    assert res_df["total_revenue"][2] == 10 * 1321
    assert res_df["total_revenue"][3] == 7 * 1255


def _nonmonotonic_num_df():
    return pd.DataFrame(
        data=[[3, 5, "c"], [1, 2, "a"], [2, 4, "b"]],
        index=["row-c", "row-a", "row-b"],
        columns=["num1", "num2", "char"],
    )


def test_applytorows_parallel_scalar_matches_serial():
    """Testing opt-in ApplyToRows parallel scalar execution."""
    df = _some_df()
    serial_stage = ApplyToRows(_total_rev, "total_revenue", n_jobs=1)
    parallel_stage = ApplyToRows(_total_rev, "total_revenue", n_jobs=2)
    pd.testing.assert_frame_equal(parallel_stage(df), serial_stage(df))


def test_applytorows_parallel_df_generation_matches_serial():
    """Testing opt-in ApplyToRows parallel row expansion."""
    df = _num_df()
    serial_stage = ApplyToRows(_sum_and_diff, n_jobs=1)
    parallel_stage = ApplyToRows(_sum_and_diff, n_jobs=2)
    pd.testing.assert_frame_equal(parallel_stage(df), serial_stage(df))


def test_applytorows_parallel_follow_column_matches_serial():
    """Testing ApplyToRows parallel follow_column behavior."""
    df = _num_df()
    serial_stage = ApplyToRows(
        _sum_and_diff,
        follow_column="num1",
        n_jobs=1,
    )
    parallel_stage = ApplyToRows(
        _sum_and_diff,
        follow_column="num1",
        n_jobs=2,
    )
    parallel_df = parallel_stage(df)
    pd.testing.assert_frame_equal(parallel_df, serial_stage(df))
    assert list(parallel_df.columns) == ["num1", "diff", "sum", "num2", "char"]


def test_applytorows_parallel_preserves_index_and_column_order():
    """Testing ApplyToRows parallel output shape invariants."""
    df = _nonmonotonic_num_df()
    parallel_stage = ApplyToRows(
        _sum_and_diff,
        follow_column="num2",
        n_jobs=2,
    )
    res_df = parallel_stage(df)
    assert list(res_df.index) == ["row-c", "row-a", "row-b"]
    assert list(res_df.columns) == ["num1", "num2", "diff", "sum", "char"]


def test_applytorows_parallel_empty_frame_matches_serial():
    """Testing ApplyToRows parallel empty-frame compatibility."""
    df = _num_df().iloc[0:0]
    serial_stage = ApplyToRows(_sum_and_diff, n_jobs=1)
    parallel_stage = ApplyToRows(_sum_and_diff, n_jobs=2)

    pd.testing.assert_frame_equal(parallel_stage(df), serial_stage(df))


def test_applytorows_n_jobs_default_none_and_one_use_serial_behavior():
    """Testing ApplyToRows default n_jobs compatibility."""
    df = _some_df()
    default_stage = ApplyToRows(_total_rev, "total_revenue")
    none_stage = ApplyToRows(_total_rev, "total_revenue", n_jobs=None)
    one_stage = ApplyToRows(_total_rev, "total_revenue", n_jobs=1)
    default_df = default_stage(df)
    pd.testing.assert_frame_equal(none_stage(df), default_df)
    pd.testing.assert_frame_equal(one_stage(df), default_df)
