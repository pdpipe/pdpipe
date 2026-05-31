"""Testing contextual runtime parameters."""

import collections

import pandas as pd
import pytest

import pdpipe as pdp
from pdpipe.core import PdpApplicationContext
from pdpipe.exceptions import PipelineApplicationError


class PutFitContextStage(pdp.PdPipelineStage):
    """A stage that stores a value in the fit context."""

    def __init__(self, key, value):
        self.key = key
        self.value = value
        super().__init__(desc="Put value in fit context")

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        self.fit_context[self.key] = self.value
        return df


class AddContextValueStage(pdp.PdPipelineStage):
    """A stage that uses a contextual constructor parameter."""

    def __init__(self, params, **kwargs):
        self.params = params
        super().__init__(desc="Add contextual value", **kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        return df.assign(result=df["a"] + self.params["offset"])


class AddContextValuesStage(pdp.PdPipelineStage):
    """A stage that sums contextual values in a constructor parameter."""

    def __init__(self, values):
        self.values = values
        super().__init__(desc="Add contextual values")

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        return df.assign(result=df["a"] + sum(self.values))


class FittableXyContextStage(pdp.PdPipelineStage):
    """A fittable X/y stage that uses the fitted transform path."""

    def __init__(self, offset):
        self.offset = offset
        super().__init__(desc="Fittable X/y contextual value")

    def _prec(self, df, y=None):
        return True

    def _fit_transform(self, df, verbose):
        return df

    def _transform(self, df, verbose):
        return df

    def _transform_Xy(self, df, y, verbose):
        return df.assign(result=df["a"] + self.offset), y + 1


def _scale_df():
    return pd.DataFrame([[2, 3], [5, 6]], columns=["a", "b"])


def test_contextual_from_application_context_in_scale():
    scale_stage = pdp.Scale(
        scaler=pdp.contextual("scaling_type", fit=False),
        joint=True,
    )
    pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(
                scaling_type=lambda df: "StandardScaler"
            ),
            scale_stage,
        ]
    )

    res = pipeline.apply(_scale_df())
    expected = pd.DataFrame(
        [[-1.264911, -0.632456], [0.632456, 1.264911]],
        columns=["a", "b"],
    )

    pd.testing.assert_frame_equal(res, expected)
    assert isinstance(
        scale_stage.scaler, pdp.run_time_parameters.ContextualParameter
    )


def test_contextual_from_fit_context_by_default_and_nested_restore():
    add_stage = AddContextValueStage({"offset": pdp.contextual("fit_offset")})
    pipeline = pdp.PdPipeline(
        [
            PutFitContextStage("fit_offset", 10),
            add_stage,
        ]
    )
    df = pd.DataFrame({"a": [1, 2]})

    res = pipeline.apply(df)

    expected = pd.DataFrame({"a": [1, 2], "result": [11, 12]})
    pd.testing.assert_frame_equal(res, expected)
    assert isinstance(
        add_stage.params["offset"], pdp.run_time_parameters.ContextualParameter
    )


def test_contextual_from_fit_context_resolves_on_later_transform():
    add_stage = AddContextValueStage({"offset": pdp.contextual("fit_offset")})
    pipeline = pdp.PdPipeline(
        [
            PutFitContextStage("fit_offset", 10),
            add_stage,
        ]
    )

    pipeline.apply(pd.DataFrame({"a": [1, 2]}))
    res = pipeline.transform(pd.DataFrame({"a": [5, 7]}))

    expected = pd.DataFrame({"a": [5, 7], "result": [15, 17]})
    pd.testing.assert_frame_equal(res, expected)


def test_contextual_application_context_resolves_on_each_transform():
    add_stage = AddContextValueStage(
        {"offset": pdp.contextual("app_offset", fit=False)}
    )
    pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(
                app_offset=lambda df: df["a"].max()
            ),
            add_stage,
        ]
    )

    first = pipeline.apply(pd.DataFrame({"a": [1, 2]}))
    second = pipeline.apply(pd.DataFrame({"a": [5, 7]}))

    pd.testing.assert_frame_equal(
        first, pd.DataFrame({"a": [1, 2], "result": [3, 4]})
    )
    pd.testing.assert_frame_equal(
        second, pd.DataFrame({"a": [5, 7], "result": [12, 14]})
    )
    assert isinstance(
        add_stage.params["offset"], pdp.run_time_parameters.ContextualParameter
    )


def test_contextual_application_context_resolves_on_timed_transform():
    add_stage = AddContextValueStage(
        {"offset": pdp.contextual("app_offset", fit=False)}
    )
    pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(
                app_offset=lambda df: df["a"].max()
            ),
            add_stage,
        ]
    )

    pipeline.apply(pd.DataFrame({"a": [1, 2]}))
    res = pipeline.transform(pd.DataFrame({"a": [5, 7]}), time=True)

    pd.testing.assert_frame_equal(
        res, pd.DataFrame({"a": [5, 7], "result": [12, 14]})
    )


def test_contextual_resolves_when_stage_is_applied_directly():
    add_stage = AddContextValueStage({"offset": pdp.contextual("fit_offset")})
    add_stage.fit_context = PdpApplicationContext()
    add_stage.fit_context.update({"fit_offset": 4})

    res = add_stage.apply(pd.DataFrame({"a": [1, 2]}))

    expected = pd.DataFrame({"a": [1, 2], "result": [5, 6]})
    pd.testing.assert_frame_equal(res, expected)


def test_contextual_runtime_transform_skip_returns_unmodified_frame():
    add_stage = AddContextValueStage(
        {"offset": 1},
        skip=lambda df: True,
    )
    df = pd.DataFrame({"a": [1, 2]})

    res = add_stage.transform(df)

    pd.testing.assert_frame_equal(res, df)


def test_contextual_runtime_transform_fitted_xy_stage():
    stage = FittableXyContextStage(3)
    df = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([10, 20])
    stage.fit(df, y=y)

    res_X, res_y = stage.transform(df, y=y)

    expected_X = pd.DataFrame({"a": [1, 2], "result": [4, 5]})
    expected_y = pd.Series([11, 21])
    pd.testing.assert_frame_equal(res_X, expected_X)
    pd.testing.assert_series_equal(res_y, expected_y)


def test_pipeline_error_message_omits_empty_error_detail():
    stage = AddContextValueStage({"offset": 1})
    msg = pdp.PdPipeline._stage_application_error_message(
        0,
        stage,
        Exception(),
    )

    assert msg == f"Exception raised in stage [ 0] {stage}"


def test_contextuals_support_mapping_attributes():
    params = collections.UserDict({"offset": pdp.contextual("fit_offset")})
    add_stage = AddContextValueStage(params)
    pipeline = pdp.PdPipeline(
        [
            PutFitContextStage("fit_offset", 8),
            add_stage,
        ]
    )

    res = pipeline.apply(pd.DataFrame({"a": [1, 2]}))

    expected = pd.DataFrame({"a": [1, 2], "result": [9, 10]})
    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "container_factory",
    [
        list,
        tuple,
        set,
    ],
)
def test_contextuals_support_sequence_attributes(container_factory):
    values = container_factory([pdp.contextual("fit_offset"), 2])
    add_stage = AddContextValuesStage(values)
    pipeline = pdp.PdPipeline(
        [
            PutFitContextStage("fit_offset", 8),
            add_stage,
        ]
    )

    res = pipeline.apply(pd.DataFrame({"a": [1, 2]}))

    expected = pd.DataFrame({"a": [1, 2], "result": [11, 12]})
    pd.testing.assert_frame_equal(res, expected)
    assert isinstance(add_stage.values, container_factory)


def test_contextuals_work_in_stage_kwargs():
    scale_stage = pdp.Scale(
        scaler="StandardScaler",
        with_mean=pdp.contextual("with_mean", fit=False),
    )
    pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(with_mean=lambda df: False),
            scale_stage,
        ]
    )

    res = pipeline.apply(pd.DataFrame({"a": [2, 5]}))

    expected = pd.DataFrame({"a": [1.333333, 3.333333]})
    pd.testing.assert_frame_equal(res, expected)
    assert isinstance(
        scale_stage._kwargs["with_mean"],
        pdp.run_time_parameters.ContextualParameter,
    )


def test_contextual_missing_key_raises_pipeline_error():
    pipeline = pdp.PdPipeline(
        [
            AddContextValueStage(
                {"offset": pdp.contextual("missing", fit=False)}
            )
        ]
    )

    with pytest.raises(
        PipelineApplicationError,
        match="missing.*application_context",
    ):
        pipeline.apply(pd.DataFrame({"a": [1, 2]}))


def test_contextual_rejects_non_bool_fit_flag():
    with pytest.raises(TypeError, match="'fit' must be a bool"):
        pdp.contextual("offset", fit=None)


def test_contextual_rejects_non_string_key():
    with pytest.raises(TypeError, match="'key' must be a str"):
        pdp.contextual(1)


def test_contextuals_work_in_timed_pipeline_path():
    add_stage = AddContextValueStage(
        {"offset": pdp.contextual("app_offset", fit=False)}
    )
    pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(app_offset=lambda df: 5),
            add_stage,
        ]
    )

    res = pipeline.apply(pd.DataFrame({"a": [1, 2]}), time=True)

    expected = pd.DataFrame({"a": [1, 2], "result": [6, 7]})
    pd.testing.assert_frame_equal(res, expected)


def test_contextuals_work_in_trace_path():
    add_stage = AddContextValueStage(
        {"offset": pdp.contextual("app_offset", fit=False)}
    )
    pipeline = pdp.PdPipeline(
        [
            pdp.ApplicationContextEnricher(app_offset=lambda df: 3),
            add_stage,
        ]
    )

    trace = pipeline.trace(pd.DataFrame({"a": [1, 2]}))

    assert trace[-1]["status"] == "applied"
    assert trace[-1]["output_columns"] == ["a", "result"]
    assert trace[-1]["output_shape"] == (2, 2)
