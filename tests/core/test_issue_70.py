"""Regression tests for issue #70 condition error messages."""

import pandas as pd
import pytest

import pdpipe as pdp
from pdpipe.exceptions import (
    FailedPostconditionError,
    FailedPreconditionError,
    PipelineApplicationError,
)


class _StageWithFailingPrecondition(pdp.PdPipelineStage):
    """Stage with an always-failing built-in precondition."""

    def __init__(self, **kwargs):
        super().__init__(exmsg="Stage precondition marker", **kwargs)

    def _prec(self, X):
        return False

    def _transform(self, X, verbose=False):
        return X


class _StageWithFailingPostcondition(pdp.PdPipelineStage):
    """Stage with an always-failing built-in postcondition."""

    def __init__(self, **kwargs):
        super().__init__(exmsg="Stage precondition marker", **kwargs)

    def _prec(self, X):
        return True

    def _post(self, X):
        return False

    def _transform(self, X, verbose=False):
        return X


def _test_df():
    return pd.DataFrame(
        [[1, 4], [4, 5], [1, 11]],
        [1, 2, 3],
        ["a", "b"],
    )


def _pipeline_cause(stage, df):
    pline = pdp.PdPipeline([stage])
    with pytest.raises(PipelineApplicationError) as exc_info:
        pline.apply(df)
    return exc_info.value.__cause__


def test_user_precondition_error_message():
    cause = _pipeline_cause(
        stage=_StageWithFailingPrecondition(
            prec=pdp.cond.HasAllColumns(["x"]),
        ),
        df=_test_df(),
    )
    assert isinstance(cause, FailedPreconditionError)
    message = str(cause)
    assert "Not all required columns x" in message
    assert "Stage precondition marker" not in message


def test_stage_precondition_error_message():
    cause = _pipeline_cause(
        stage=_StageWithFailingPrecondition(
            prec=pdp.cond.HasAllColumns(["a"]),
        ),
        df=_test_df(),
    )
    assert isinstance(cause, FailedPreconditionError)
    assert str(cause) == "Stage precondition marker"


def test_user_postcondition_error_message():
    cause = _pipeline_cause(
        stage=_StageWithFailingPostcondition(
            post=pdp.cond.HasAllColumns(["x"]),
        ),
        df=_test_df(),
    )
    assert isinstance(cause, FailedPostconditionError)
    message = str(cause)
    assert "Not all required columns x" in message
    assert "Stage postcondition marker" not in message


def test_stage_postcondition_error_message():
    cause = _pipeline_cause(
        stage=_StageWithFailingPostcondition(
            post=pdp.cond.HasAllColumns(["a"]),
        ),
        df=_test_df(),
    )
    assert isinstance(cause, FailedPostconditionError)
    assert str(cause) == "Stage postcondition marker"
