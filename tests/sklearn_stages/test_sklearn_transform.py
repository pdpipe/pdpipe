"""Testing sklearn_stages.SklearnColumnTransform."""

import numpy as np
import pandas as pd
import pytest

import pdpipe as pdp
from pdpipe.exceptions import PipelineApplicationError
from pdpipe.sklearn_stages import SklearnColumnTransform

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _some_df():
    return pd.DataFrame(
        data=[
            ["r1", 1.0, 10.0, 100.0, "a"],
            ["r2", 2.0, 20.0, 200.0, "b"],
            ["r3", 3.0, 30.0, 300.0, "c"],
        ],
        index=[1, 2, 3],
        columns=["id", "x", "y", "z", "label"],
    )


def test_sklearn_transform_single_column_selector():
    df = _some_df()
    stage = SklearnColumnTransform(StandardScaler(), "x")

    res = stage(df)

    assert list(res.columns) == ["id", "x", "y", "z", "label"]
    assert res["id"].equals(df["id"])
    assert res["y"].equals(df["y"])
    assert res["label"].equals(df["label"])
    assert np.isclose(res["x"].mean(), 0)


def test_sklearn_transform_list_selector_and_fit_transform_consistency():
    df = _some_df()
    df2 = pd.DataFrame(
        data=[
            ["s1", 4.0, 40.0, 400.0, "d"],
            ["s2", 5.0, 50.0, 500.0, "e"],
            ["s3", 6.0, 60.0, 600.0, "f"],
        ],
        index=[4, 5, 6],
        columns=df.columns,
    )
    stage = SklearnColumnTransform(StandardScaler(), ["x", "y"])
    stage(df)

    res = stage(df2)
    expected = (
        StandardScaler()
        .fit(df[["x", "y"]].values)
        .transform(df2[["x", "y"]].values)
    )

    assert list(res.columns) == ["id", "x", "y", "z", "label"]
    assert np.allclose(res[["x", "y"]].values, expected)
    assert res["id"].equals(df2["id"])
    assert res["z"].equals(df2["z"])
    assert res["label"].equals(df2["label"])


def test_sklearn_transform_column_qualifier_selector():
    df = _some_df()
    stage = SklearnColumnTransform(StandardScaler(), pdp.cq.StartsWith("x"))

    res = stage(df)

    assert list(res.columns) == ["id", "x", "y", "z", "label"]
    assert np.isclose(res["x"].mean(), 0)
    assert res["y"].equals(df["y"])


def test_sklearn_transform_callable_selector():
    df = _some_df()
    stage = SklearnColumnTransform(
        StandardScaler(),
        lambda X: [col for col in X.columns if col in ["y", "z"]],
    )

    res = stage(df)

    assert list(res.columns) == ["id", "x", "y", "z", "label"]
    assert res["x"].equals(df["x"])
    assert np.isclose(res["y"].mean(), 0)
    assert np.isclose(res["z"].mean(), 0)


def test_sklearn_transform_shape_changing_generated_columns():
    df = _some_df()
    stage = SklearnColumnTransform(
        PCA(n_components=1),
        ["x", "y", "z"],
        lbl_format="pca{}",
    )

    res = stage(df)

    assert list(res.columns) == ["id", "pca0", "label"]
    assert len(res.columns) == 3
    assert res["id"].equals(df["id"])
    assert res["label"].equals(df["label"])


def test_sklearn_transform_shape_changing_result_columns():
    df = _some_df()
    stage = SklearnColumnTransform(
        PCA(n_components=2),
        ["x", "y", "z"],
        result_columns=["comp_a", "comp_b"],
    )

    res = stage(df)

    assert list(res.columns) == ["id", "comp_a", "comp_b", "label"]
    assert res["id"].equals(df["id"])
    assert res["label"].equals(df["label"])


def test_sklearn_transform_noncontiguous_same_width_columns_keep_positions():
    df = _some_df()
    stage = SklearnColumnTransform(
        FunctionTransformer(lambda X: X + 1),
        ["x", "z"],
    )

    res = stage(df)

    assert list(res.columns) == ["id", "x", "y", "z", "label"]
    assert res["y"].equals(df["y"])
    assert np.allclose(res[["x", "z"]].values, df[["x", "z"]].values + 1)


def test_sklearn_transform_same_width_custom_columns_keep_positions():
    df = _some_df()
    stage = SklearnColumnTransform(
        FunctionTransformer(lambda X: X + 1),
        ["x", "z"],
        result_columns=["x_plus", "z_plus"],
    )

    res = stage(df)

    assert list(res.columns) == ["id", "x_plus", "y", "z_plus", "label"]
    assert res["y"].equals(df["y"])
    assert np.allclose(
        res[["x_plus", "z_plus"]].values, df[["x", "z"]].values + 1
    )


def test_sklearn_transform_invalid_result_columns_length():
    df = _some_df()
    stage = SklearnColumnTransform(
        PCA(n_components=1),
        ["x", "y", "z"],
        result_columns=["a", "b"],
    )

    with pytest.raises(
        PipelineApplicationError, match="result_columns length"
    ):
        stage(df)


def test_sklearn_transform_rejects_non_2d_output():
    df = _some_df()
    stage = SklearnColumnTransform(
        FunctionTransformer(lambda X: X.iloc[:, 0]),
        ["x", "y"],
    )

    with pytest.raises(PipelineApplicationError, match="2-dimensional"):
        stage(df)


class _ShortOutputTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:-1]


def test_sklearn_transform_rejects_wrong_row_count():
    df = _some_df()
    stage = SklearnColumnTransform(_ShortOutputTransformer(), ["x", "y"])

    with pytest.raises(PipelineApplicationError, match="row count"):
        stage(df)


class _DataFrameAwareTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == self.columns_
        return X + 1


def test_sklearn_transform_passes_sub_dataframe_to_transformer():
    df = _some_df()
    stage = SklearnColumnTransform(_DataFrameAwareTransformer(), ["x", "y"])

    res = stage(df)

    assert list(res.columns) == ["id", "x", "y", "z", "label"]
    assert np.allclose(res[["x", "y"]].values, df[["x", "y"]].values + 1)


class _ChangingWidthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.index) == 3:
            return X.iloc[:, [0]]
        return X


def test_sklearn_transform_rejects_transform_width_change():
    df = _some_df()
    df2 = pd.DataFrame(
        data=[
            ["s1", 4.0, 40.0, 400.0, "d"],
            ["s2", 5.0, 50.0, 500.0, "e"],
        ],
        index=[4, 5],
        columns=df.columns,
    )
    stage = SklearnColumnTransform(_ChangingWidthTransformer(), ["x", "y"])
    stage(df2)

    with pytest.raises(PipelineApplicationError, match="width changed"):
        stage(df)


def test_sklearn_transform_missing_fitted_column_raises_pipeline_error():
    df = _some_df()
    stage = SklearnColumnTransform(
        StandardScaler(),
        lambda X: [col for col in ["x", "y"] if col in X.columns],
    )
    stage(df)

    with pytest.raises(PipelineApplicationError, match="missing fitted"):
        stage(df.drop(columns=["y"]))


def test_sklearn_transform_rejects_result_column_collision():
    df = _some_df()
    stage = SklearnColumnTransform(
        PCA(n_components=1),
        ["x", "y"],
        result_columns=["label"],
    )

    with pytest.raises(PipelineApplicationError, match="collide"):
        stage(df)


def test_sklearn_missing_dep_sklearn_transform():
    import pdpipe.sklearn_stages as sk

    original = sk._SKLEARN_INSTALLED
    try:
        sk._SKLEARN_INSTALLED = False
        with pytest.raises(ImportError, match="scikit-learn is required"):
            SklearnColumnTransform(StandardScaler(), "x")
    finally:
        sk._SKLEARN_INSTALLED = original
