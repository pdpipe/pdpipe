"""Testing basic pipeline stages."""

from builtins import ValueError

import pandas as pd
import pytest

from pdpipe.core import PdPipelineStage, PdPipeline
from pdpipe import make_pdpipeline, ColByFrameFunc, ColDrop
from pdpipe.exceptions import PipelineApplicationError


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, "a"], [2, 4, "b"]],
        index=[1, 2],
        columns=["num1", "num2", "char"],
    )


class SilentDropStage(PdPipelineStage):
    """A pipeline stage for testing."""

    def __init__(self, colname, **kwargs):
        self.colname = colname
        super().__init__(exraise=False, **kwargs)

    def _prec(self, df):
        return self.colname in df.columns

    def _transform(self, df, verbose):
        return df.drop([self.colname], axis=1)


class FailingTransformStage(PdPipelineStage):
    """A pipeline stage that fails during transformation."""

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        raise ValueError("trace failure")


class MutatingStage(PdPipelineStage):
    """A pipeline stage that mutates its input dataframe."""

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        df["mutated"] = 1
        return df


class FittableMarkingStage(PdPipelineStage):
    """A fittable pipeline stage for trace isolation tests."""

    def _prec(self, df):
        return True

    def _fit_transform(self, df, verbose):
        self.was_fitted = True
        return df.drop(["num1"], axis=1)

    def _transform(self, df, verbose):
        return df.drop(["num1"], axis=1)


@pytest.mark.parametrize("time", [True, False])
def test_two_stage_pipeline_stage(time):
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    pipeline = PdPipeline([drop_num1, drop_num2])
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns
    str(pipeline)
    pipeline.memory_report()

    # test fit
    df = _test_df()
    res_df = pipeline.fit(df, verbose=True, time=time)
    for x in ["num1", "num2", "char"]:
        assert x in res_df.columns

    # test apply
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True, time=time)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns

    # test transform
    df = _test_df()
    res_df = pipeline.transform(df, verbose=True, time=time)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns

    # test fit_transform
    df = _test_df()
    res_df = pipeline.fit_transform(df, verbose=True, time=time)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns

    # test get_transformer
    trs = lambda pipline: pipeline[:1]  # noqa: E731
    pipeline = PdPipeline([drop_num1, drop_num2], transformer_getter=trs)
    transformer = pipeline.get_transformer()
    res_df = transformer(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" in res_df.columns
    assert "char" in res_df.columns


def test_make_pdpipeline():
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    pipeline = make_pdpipeline(drop_num1, drop_num2)
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


def test_pipeline_stage_addition():
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    pipeline = drop_num1 + drop_num2
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


def test_pipeline_to_pipeline_stage_addition():
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    pipeline = PdPipeline([drop_num1])
    assert len(pipeline) == 1
    pipeline = pipeline + drop_num2
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


def test_pipeline_stage_to_pipeline_addition():
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    pipeline = PdPipeline([drop_num1])
    assert len(pipeline) == 1
    pipeline = drop_num2 + pipeline
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


def test_pipeline_to_pipeline_addition():
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    pipeline1 = PdPipeline([drop_num1])
    pipeline2 = PdPipeline([drop_num2])
    pipeline = pipeline1 + pipeline2
    assert len(pipeline) == 2
    assert pipeline[0] == drop_num1
    assert pipeline[1] == drop_num2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


def test_pipeline_to_int_addition():
    """Testing something."""
    pipeline = PdPipeline([SilentDropStage("num1")])
    with pytest.raises(TypeError):
        res = pipeline + 43
        assert not isinstance(res, PdPipeline)


def test_pipeline_to_dot_represents_stage_order_and_names():
    """Test pipeline DOT export uses stable ordered nodes."""
    drop_num1 = SilentDropStage(
        "num1",
        name="drop_num1",
        desc="Drop num1 column",
    )
    drop_num2 = SilentDropStage("num2", desc="Drop num2 column")
    pipeline = PdPipeline([drop_num1, drop_num2])

    assert pipeline.to_dot() == "\n".join(
        [
            "digraph PdPipeline {",
            "  graph [rankdir=LR];",
            "  node [shape=box];",
            (
                '  stage_0 [label="[0] SilentDropStage: '
                'drop_num1\\nDrop num1 column"];'
            ),
            '  stage_1 [label="[1] SilentDropStage\\nDrop num2 column"];',
            "  stage_0 -> stage_1;",
            "}",
        ]
    )


def test_pipeline_to_dot_escapes_dot_label_characters():
    """Test pipeline DOT export escapes label text."""
    stage = SilentDropStage(
        "num1",
        name='drop "quoted"',
        desc='Drop "quoted" \\ column\nthen continue',
    )
    pipeline = PdPipeline([stage])

    assert pipeline.to_dot() == "\n".join(
        [
            "digraph PdPipeline {",
            "  graph [rankdir=LR];",
            "  node [shape=box];",
            (
                '  stage_0 [label="[0] SilentDropStage: drop \\"quoted\\"\\n'
                'Drop \\"quoted\\" \\\\ column\\nthen continue"];'
            ),
            "}",
        ]
    )


def test_pipeline_to_dot_is_deterministic_across_calls():
    """Test pipeline DOT export is deterministic."""
    pipeline = PdPipeline(
        [
            SilentDropStage("num1", desc="Drop num1 column"),
            SilentDropStage("num2", desc="Drop num2 column"),
        ]
    )

    assert pipeline.to_dot() == pipeline.to_dot()


def test_empty_pipeline_to_dot():
    """Test DOT export for empty pipelines."""
    pipeline = PdPipeline([])

    assert pipeline.to_dot() == "\n".join(
        [
            "digraph PdPipeline {",
            "  graph [rankdir=LR];",
            "  node [shape=box];",
            "}",
        ]
    )


def test_pipeline_trace_reports_applied_and_precondition_skipped_stages():
    """Test structured trace records for applied and precondition skips."""
    drop_num1 = SilentDropStage(
        "num1",
        name="drop_num1",
        desc="Drop num1 column",
    )
    missing = SilentDropStage("missing", desc="Drop missing column")
    pipeline = PdPipeline([drop_num1, missing])
    df = _test_df()

    trace = pipeline.trace(df)

    assert trace == [
        {
            "stage_index": 0,
            "stage_class": "SilentDropStage",
            "stage_name": "drop_num1",
            "stage_description": "Drop num1 column",
            "status": "applied",
            "skip_reason": None,
            "input_shape": (2, 3),
            "input_columns": ["num1", "num2", "char"],
            "output_shape": (2, 2),
            "output_columns": ["num2", "char"],
            "error_type": None,
            "error_message": None,
        },
        {
            "stage_index": 1,
            "stage_class": "SilentDropStage",
            "stage_name": "",
            "stage_description": "Drop missing column",
            "status": "skipped",
            "skip_reason": "precondition",
            "input_shape": (2, 2),
            "input_columns": ["num2", "char"],
            "output_shape": (2, 2),
            "output_columns": ["num2", "char"],
            "error_type": None,
            "error_message": None,
        },
    ]


def test_pipeline_trace_reports_skip_callable_stages():
    """Test trace distinguishes skip callbacks from precondition skips."""
    stage = SilentDropStage(
        "num1",
        skip=lambda df: True,
        desc="Conditionally drop num1",
    )
    pipeline = PdPipeline([stage])

    trace = pipeline.trace(_test_df())

    assert trace[0]["status"] == "skipped"
    assert trace[0]["skip_reason"] == "skip"
    assert trace[0]["input_columns"] == ["num1", "num2", "char"]
    assert trace[0]["output_columns"] == ["num1", "num2", "char"]


def test_pipeline_trace_reports_failure_and_stops():
    """Test trace records a failing stage and stops at the failure."""
    pipeline = PdPipeline(
        [
            SilentDropStage("num1"),
            FailingTransformStage(desc="Fail in transform"),
            SilentDropStage("char"),
        ]
    )

    trace = pipeline.trace(_test_df())

    assert [entry["status"] for entry in trace] == ["applied", "failed"]
    assert trace[1]["stage_index"] == 1
    assert trace[1]["stage_class"] == "FailingTransformStage"
    assert trace[1]["stage_description"] == "Fail in transform"
    assert trace[1]["input_shape"] == (2, 2)
    assert trace[1]["input_columns"] == ["num2", "char"]
    assert trace[1]["output_shape"] is None
    assert trace[1]["output_columns"] is None
    assert trace[1]["error_type"] == "ValueError"
    assert trace[1]["error_message"] == "trace failure"


def test_pipeline_trace_is_deterministic_across_calls():
    """Test trace output order and content are deterministic."""
    pipeline = PdPipeline(
        [
            SilentDropStage("num1", name="first"),
            SilentDropStage("num2", name="second"),
            SilentDropStage("char", name="third"),
        ]
    )
    df = _test_df()

    first_trace = pipeline.trace(df)
    second_trace = pipeline.trace(df)

    assert first_trace == second_trace
    assert [entry["stage_index"] for entry in first_trace] == [0, 1, 2]
    assert [entry["stage_name"] for entry in first_trace] == [
        "first",
        "second",
        "third",
    ]


def test_pipeline_trace_does_not_mutate_input_dataframe_or_pipeline_state():
    """Test trace works on copies of both the dataframe and pipeline."""
    fittable = FittableMarkingStage()
    pipeline = PdPipeline([fittable, MutatingStage()])
    df = _test_df()
    expected_df = df.copy(deep=True)

    trace = pipeline.trace(df)

    pd.testing.assert_frame_equal(df, expected_df)
    assert "mutated" not in df.columns
    assert not pipeline.is_fitted
    assert not fittable.is_fitted
    assert not hasattr(fittable, "was_fitted")
    assert trace[0]["status"] == "applied"
    assert trace[0]["output_columns"] == ["num2", "char"]
    assert trace[1]["status"] == "applied"
    assert trace[1]["input_columns"] == ["num2", "char"]
    assert trace[1]["output_columns"] == ["num2", "char", "mutated"]


def test_pipeline_index():
    """Testing something."""
    df = _test_df()
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    drop_char = SilentDropStage("char")
    pipeline = PdPipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    assert pipeline[0] == drop_num1
    assert "num1" not in pipeline[0](df).columns
    assert pipeline[1] == drop_num2
    assert "num2" not in pipeline[1](df).columns
    assert pipeline[2] == drop_char
    assert "char" not in pipeline[2](df).columns


def test_pipeline_slice():
    """Testing something."""
    drop_num1 = SilentDropStage("num1")
    drop_num2 = SilentDropStage("num2")
    drop_char = SilentDropStage("char")
    pipeline = PdPipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    pipeline = pipeline[0:2]
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


def test_pipeline_slice_by_name():
    """Testing something."""
    drop_num1 = SilentDropStage("num1", name="dropNum1")
    drop_num2 = SilentDropStage("num2", name="dropNum2")
    drop_char = SilentDropStage("char", name="dropChar")
    pipeline = PdPipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    pipeline = pipeline[["dropNum1", "dropNum2"]]
    assert len(pipeline) == 2
    assert pipeline["dropNum1"] == drop_num1
    with pytest.raises(ValueError) as e:
        pipeline["dropChar"]
    assert str(e.value) == "'dropChar' is not exist."
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert "num1" not in res_df.columns
    assert "num2" not in res_df.columns
    assert "char" in res_df.columns


@pytest.mark.parametrize("time", [True, False])
def test_pipeline_error(time):
    """Test exceptions at pipeline level."""

    # test fit
    df = _test_df()

    def _func(df):
        return df["num1"] == df["num3"]

    pipeline = PdPipeline([ColByFrameFunc("Equality", _func), ColDrop("B")])
    with pytest.raises(PipelineApplicationError):
        pipeline.fit(df, verbose=True, time=time)

    # test apply
    df = _test_df()
    with pytest.raises(PipelineApplicationError):
        pipeline.apply(df, verbose=True, time=time)

    # test transform
    df = _test_df()
    with pytest.raises(PipelineApplicationError):
        pipeline.transform(df, verbose=True, time=time)

    # test fit_transform
    df = _test_df()
    with pytest.raises(PipelineApplicationError):
        pipeline.fit_transform(df, verbose=True, time=time)
