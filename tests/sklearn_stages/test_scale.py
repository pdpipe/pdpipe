"""Testing basic pipeline stages."""

import pytest
import pandas as pd
# from numpy.testing import assert_approx_equal

from pdpipe.sklearn_stages import Scale
from pdpipe.exceptions import PipelineApplicationError


def _some_df1():
    return pd.DataFrame(
        data=[[3.2, 0.3, "A"], [7.2, 0.35, "B"], [12.1, 0.29, "C"]],
        index=[1, 2, 3],
        columns=["ph", "gt", "lbl"],
    )


def _some_df1b():
    return pd.DataFrame(
        data=[[3.8, 0.45, "A"], [7.7, 0.31, "B"], [11.15, 0.33, "C"]],
        index=[1, 2, 3],
        columns=["ph", "gt", "lbl"],
    )


def _bad_dtype_df1():
    return pd.DataFrame(
        data=[[3.2, "H", "A"], [7.2, "I", "B"], [12.1, "P", "C"]],
        index=[1, 2, 3],
        columns=["ph", "gt", "lbl"],
    )


def _some_df2():
    return pd.DataFrame(
        data=[[3.2, 0.3], [7.2, 0.35], [12.1, 0.29]],
        index=[1, 2, 3],
        columns=["ph", "gt"],
    )


def _some_df2b():
    return pd.DataFrame(
        data=[[3.8, 0.45], [7.7, 0.31], [11.15, 0.33]],
        index=[1, 2, 3],
        columns=["ph", "gt"],
    )


def test_scale():
    df = _some_df2()
    scale_stage = Scale("StandardScaler")
    res_df = scale_stage(df)
    assert "ph" in res_df.columns
    assert "gt" in res_df.columns
    assert res_df["ph"][1] < df["ph"][1]

    # see only transform (no fit) when already fitted
    df2 = _some_df2b()
    res_df2 = scale_stage(df2)
    assert "ph" in res_df2.columns
    assert "gt" in res_df2.columns
    assert res_df2["ph"][1] < df2["ph"][1]
    assert res_df["ph"][1] < res_df2["ph"][1]

    # check fit_transform when already fitted
    df3 = _some_df2b()
    res_df3 = scale_stage.fit_transform(df2)
    assert "ph" in res_df3.columns
    assert "gt" in res_df3.columns
    assert res_df3["ph"][1] < df3["ph"][1]
    assert res_df3["ph"][1] < res_df2["ph"][1]


def test_scale_with_exclude_cols():
    df = _some_df1()
    scale_stage = Scale("StandardScaler", exclude_columns=["lbl"])
    res_df = scale_stage(df)
    assert list(res_df.columns) == ["ph", "gt", "lbl"]
    assert "ph" in res_df.columns
    assert "gt" in res_df.columns
    assert res_df["ph"][1] < df["ph"][1]

    # see only transform (no fit) when already fitted
    df2 = _some_df1b()
    res_df2 = scale_stage(df2)
    assert "ph" in res_df2.columns
    assert "gt" in res_df2.columns
    assert df['ph'][1] < df2['ph'][1]
    assert res_df2["ph"][1] < df2["ph"][1]
    assert res_df["ph"][1] < res_df2["ph"][1]

    # check fit_transform when already fitted
    df3 = _some_df1b()
    res_df3 = scale_stage.fit_transform(df2)
    assert "ph" in res_df3.columns
    assert "gt" in res_df3.columns
    assert res_df3["ph"][1] < df3["ph"][1]
    assert res_df3["ph"][1] < res_df2["ph"][1]


def test_scale_with_exclude():
    """Basic binning test."""
    df = _some_df2()
    scale_stage = Scale("StandardScaler", with_std=False)
    res_df = scale_stage(df)
    assert "ph" in res_df.columns
    assert "gt" in res_df.columns


def test_scale_fit_transform_exception():
    df1 = _some_df1()
    scale_stage = Scale("StandardScaler", columns=['ph', 'lbl'])
    with pytest.raises(PipelineApplicationError):
        scale_stage(df1)


def test_scale_transform_exception():
    df1 = _some_df1()
    scale_stage = Scale("StandardScaler", exmsg="ERR")
    scale_stage(df1)

    # test transform exception
    df2 = _bad_dtype_df1()
    with pytest.raises(PipelineApplicationError):
        scale_stage(df2)


def _some_df3():
    return pd.DataFrame(
        data=[[1, 1], [2, 4]],
        index=[1, 2],
        columns=["a", "b"],
    )


def _some_df3b():
    return pd.DataFrame(
        data=[[1, 1], [2, 3]],
        index=[1, 2],
        columns=["a", "b"],
    )


def test_scale_with_joint():
    df = _some_df3()
    scale_stage = Scale("MinMaxScaler")
    res_df = scale_stage(df)
    assert "a" in res_df.columns
    assert "b" in res_df.columns
    assert res_df["a"][2] == 1
    assert res_df["b"][2] == 1

    df = _some_df3()
    scale_stage = Scale("MinMaxScaler", joint=True)
    res_df = scale_stage(df)
    assert "a" in res_df.columns
    assert "b" in res_df.columns
    assert res_df["a"][2] == (1 / 3)
    assert res_df["b"][2] == 1

    df = _some_df3b()
    res_df = scale_stage(df)
    assert "a" in res_df.columns
    assert "b" in res_df.columns
    assert (res_df >= 1).sum().sum() == 0
