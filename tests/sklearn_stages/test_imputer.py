"""Testing Imputer pipeline stage."""

import pickle

import pandas as pd
import numpy as np

from pdpipe.exceptions import PipelineApplicationError
from pdpipe.sklearn_stages import Imputer

from pdptestutil import random_pickle_path


def _some_df_with_nans():
    return pd.DataFrame(
        data=[[1.0, np.nan, "A"], [2.0, 4.0, "B"], [np.nan, 6.0, "C"]],
        index=[1, 2, 3],
        columns=["x", "y", "lbl"],
    )


def _some_df_with_nans_all_cols():
    return pd.DataFrame(
        data=[[np.nan, np.nan], [2.0, 4.0], [np.nan, 6.0]],
        index=[1, 2, 3],
        columns=["x", "y"],
    )


def _some_df_with_nans_b():
    return pd.DataFrame(
        data=[[3.0, np.nan, "D"], [5.0, 8.0, "E"], [np.nan, 10.0, "F"]],
        index=[1, 2, 3],
        columns=["x", "y", "lbl"],
    )


def _some_df_no_nans():
    return pd.DataFrame(
        data=[[1.0, 2.0, "A"], [3.0, 4.0, "B"], [5.0, 6.0, "C"]],
        index=[1, 2, 3],
        columns=["x", "y", "lbl"],
    )


def test_imputer_mean():
    """Test imputation with mean strategy."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("mean", columns=["x", "y"])
    res_df = imputer_stage(df)

    # Check that NaNs are imputed
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0

    # Check column names preserved
    assert list(res_df.columns) == ["x", "y", "lbl"]

    # Check that mean imputation occurred correctly
    # mean of [1.0, 2.0, nan] = 1.5
    assert res_df["x"][3] == 1.5
    # mean of [nan, 4.0, 6.0] = 5.0
    assert res_df["y"][1] == 5.0


def test_imputer_mean_transform():
    """Test imputation transform with mean strategy."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("mean", columns=["x", "y"])
    imputer_stage(df)

    # Apply to new data using fitted imxxxxr
    df2 = _some_df_with_nans_b()
    res_df2 = imputer_stage(df2)

    # Check that NaNs are imputed in new data
    assert res_df2["x"].isna().sum() == 0
    assert res_df2["y"].isna().sum() == 0

    # The imputation should use the mean from the first fit
    # mean of x from df = 1.5, so NaN at index 3 in df2 should become 1.5
    assert res_df2["x"][3] == 1.5


def test_imputer_median():
    """Test imputation with median strategy."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("median", columns=["x", "y"])
    res_df = imputer_stage(df)

    # Check that NaNs are imputed
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0

    # Check column names preserved
    assert list(res_df.columns) == ["x", "y", "lbl"]

    # Check that median imputation occurred correctly
    # median of [1.0, 2.0, nan] = 1.5
    assert res_df["x"][3] == 1.5
    # median of [nan, 4.0, 6.0] = 5.0
    assert res_df["y"][1] == 5.0


def test_imputer_with_exclude_cols():
    """Test imputation with excluded columns."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("mean", columns=["x", "y"], exclude_columns=["y"])
    res_df = imputer_stage(df)

    # Check that only x is imputed (y is excluded)
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 1  # y should still have NaN

    # Check column order preserved
    assert list(res_df.columns) == ["x", "y", "lbl"]


def test_imputer_all_columns():
    """Test imputation with all numeric columns."""
    df = _some_df_with_nans_all_cols()
    imputer_stage = Imputer("mean", columns=["x", "y"])
    res_df = imputer_stage(df)

    # Check that NaNs are imputed in all columns
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0


def test_imputer_defaults_to_all_columns():
    """Test default columns behavior."""
    df = _some_df_with_nans_all_cols()
    imputer_stage = Imputer("mean")
    res_df = imputer_stage(df)

    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0


def test_imputer_transform_with_only_selected_columns():
    """Test transform path with no unimputed columns."""
    df = _some_df_with_nans_all_cols()
    imputer_stage = Imputer("mean", columns=["x", "y"])
    imputer_stage(df)
    res_df = imputer_stage(df)

    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0
    assert list(res_df.columns) == ["x", "y"]


def test_imputer_kwargs_split_between_stage_and_imputer():
    """Test that stage kwargs are consumed and imputer kwargs remain."""
    imputer_stage = Imputer(
        "mean",
        columns=["x", "y"],
        exmsg="custom message",
        missing_values=np.nan,
    )
    assert "exmsg" not in imputer_stage._kwargs
    assert imputer_stage._kwargs["missing_values"] is np.nan


def test_imputer_fit_failure_raises_pipeline_application_error():
    """Test fit failure handling."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("mean", columns=["x", "lbl"])

    try:
        imputer_stage(df)
        assert False, "Expected PipelineApplicationError"
    except PipelineApplicationError:
        assert True


def test_imputer_transform_failure_raises_pipeline_application_error():
    """Test transform failure handling."""
    fit_df = _some_df_with_nans_all_cols()
    transform_df = pd.DataFrame(
        data=[["A", np.nan], ["B", 4.0], ["C", 6.0]],
        index=[1, 2, 3],
        columns=["x", "y"],
    )
    imputer_stage = Imputer("mean", columns=["x", "y"])
    imputer_stage(fit_df)

    try:
        imputer_stage(transform_df)
        assert False, "Expected PipelineApplicationError"
    except PipelineApplicationError:
        assert True


def test_imputer_constant_strategy():
    """Test imputation with constant strategy."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("constant", columns=["x", "y"], fill_value=0)
    res_df = imputer_stage(df)

    # Check that NaNs are imputed with constant value
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0

    # Check that NaN values were replaced with fill_value
    assert res_df["x"][3] == 0
    assert res_df["y"][1] == 0


def test_imputer_most_frequent():
    """Test imputation with most_frequent strategy."""
    df = pd.DataFrame(
        data=[[1.0, np.nan], [1.0, 2.0], [np.nan, 2.0]],
        index=[1, 2, 3],
        columns=["x", "y"],
    )
    imputer_stage = Imputer("most_frequent", columns=["x", "y"])
    res_df = imputer_stage(df)

    # Check that NaNs are imputed
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 0


def test_imputer_no_nans():
    """Test imputation on data with no NaNs."""
    df = _some_df_no_nans()
    imputer_stage = Imputer("mean", columns=["x", "y"])
    res_df = imputer_stage(df)

    # Check that data is unchanged
    pd.testing.assert_frame_equal(res_df, df)


def test_imputer_fit_transform():
    """Test fit_transform on same data."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("mean", columns=["x", "y"])

    # First fit_transform
    res_df1 = imputer_stage.fit_transform(df)
    assert res_df1["x"].isna().sum() == 0

    # Second fit_transform with new data
    df2 = _some_df_with_nans_b()
    res_df2 = imputer_stage.fit_transform(df2)
    assert res_df2["x"].isna().sum() == 0

    # Values should be different because it refitted on df2
    assert res_df2["x"][3] != res_df1["x"][3]


def test_imputer_with_single_column():
    """Test imputation with a single column."""
    df = _some_df_with_nans()
    imputer_stage = Imputer("mean", columns="x")
    res_df = imputer_stage(df)

    # Check that only specified column is imputed
    assert res_df["x"].isna().sum() == 0
    assert res_df["y"].isna().sum() == 1  # y should still have NaN
    assert list(res_df.columns) == ["x", "y", "lbl"]


def test_pickle_imputer(pdpipe_tests_dir_path):
    """Testing Imputer pickling."""
    df = _some_df_with_nans()
    stage = Imputer("mean", columns=["x", "y"])
    stage(df)
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    df2 = _some_df_with_nans_b()
    res_df2 = loaded_stage(df2)
    assert res_df2["x"].isna().sum() == 0
    assert res_df2["y"].isna().sum() == 0


def test_sklearn_missing_dep_imputer():
    """Test Imputer raises ImportError when sklearn is not installed."""
    import pytest
    import pdpipe.sklearn_stages as sk

    original = sk._SKLEARN_INSTALLED
    try:
        sk._SKLEARN_INSTALLED = False
        with pytest.raises(ImportError, match="scikit-learn is required"):
            Imputer("mean")
    finally:
        sk._SKLEARN_INSTALLED = original
