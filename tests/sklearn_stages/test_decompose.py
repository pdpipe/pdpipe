"""Testing basic pipeline stages."""

import pytest
import pandas as pd
# from numpy.testing import assert_approx_equal

from pdpipe.sklearn_stages import Decompose
from pdpipe.exceptions import PipelineApplicationError

from sklearn.decomposition import PCA


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


def test_decompose():
    df = _some_df2()
    decomp_stage = Decompose(PCA(), n_components=2)
    res_df = decomp_stage(df)
    assert "ph" not in res_df.columns
    assert "gt" not in res_df.columns
    assert "mdc0" in res_df.columns
    assert "mdc1" in res_df.columns

    # see only transform (no fit) when already fitted
    df2 = _some_df2b()
    res_df2 = decomp_stage(df2)
    assert "ph" not in res_df2.columns
    assert "gt" not in res_df2.columns
    assert "mdc0" in res_df2.columns
    assert "mdc1" in res_df2.columns

    # check fit_transform when already fitted
    df3 = _some_df2b()
    res_df3 = decomp_stage.fit_transform(df3)
    assert "ph" not in res_df3.columns
    assert "gt" not in res_df3.columns
    assert "mdc0" in res_df3.columns
    assert "mdc1" in res_df3.columns


def test_decompose_with_exclude_cols():
    df = _some_df1()
    decomp_stage = Decompose(PCA(), n_components=2, exclude_columns=["lbl"])
    res_df = decomp_stage(df)
    assert list(res_df.columns) == ["lbl", "mdc0", "mdc1"]
    assert "ph" not in res_df.columns
    assert "gt" not in res_df.columns

    # see only transform (no fit) when already fitted
    df2 = _some_df1b()
    res_df2 = decomp_stage(df2)
    assert "lbl" in res_df2.columns
    assert "ph" not in res_df2.columns
    assert "gt" not in res_df2.columns
    assert "mdc0" in res_df2.columns
    assert "mdc1" in res_df2.columns


def test_decompose_without_drop():
    """Basic binning test."""
    df = _some_df1()
    decomp_stage = Decompose(PCA(), n_components=2, drop=False)
    res_df = decomp_stage(df)
    assert list(res_df.columns) == ["ph", "gt", "lbl", "mdc0", "mdc1"]

    # see only transform (no fit) when already fitted
    df2 = _some_df1b()
    res_df2 = decomp_stage(df2)
    assert list(res_df2.columns) == ["ph", "gt", "lbl", "mdc0", "mdc1"]


def test_decompose_fit_transform_exception():
    df1 = _some_df1()
    decomp_stage = Decompose(PCA(), n_components=2, columns=['ph', 'lbl'])
    with pytest.raises(PipelineApplicationError):
        decomp_stage(df1)


def test_decompose_transform_exception():
    df1 = _some_df1()
    decomp_stage = Decompose(PCA(), n_components=2, exmsg="ERR")
    decomp_stage(df1)

    # test transform exception
    df2 = _bad_dtype_df1()
    with pytest.raises(PipelineApplicationError):
        decomp_stage(df2)


def test_decompose_with_lbl_format():
    df = _some_df2()
    decomp_stage = Decompose(PCA(), lbl_format='pca{}', n_components=2)
    res_df = decomp_stage(df)
    assert "ph" not in res_df.columns
    assert "gt" not in res_df.columns
    assert "pca0" in res_df.columns
    assert "pca1" in res_df.columns

    # see only transform (no fit) when already fitted
    df2 = _some_df2b()
    res_df2 = decomp_stage(df2)
    assert "ph" not in res_df2.columns
    assert "gt" not in res_df2.columns
    assert "pca0" in res_df2.columns
    assert "pca1" in res_df2.columns
