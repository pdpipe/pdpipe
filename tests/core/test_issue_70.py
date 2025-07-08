"""Test for issue #70: Incorrect precondition and postcondition error messages.

This module tests that user-provided preconditions/postconditions are properly
distinguished from stage preconditions/postconditions in error messages.

"""

import pandas as pd
import pytest

import pdpipe as pdp


def _test_df_with_columns(columns):
    """Helper to create test DataFrame with specified columns."""
    return pd.DataFrame([[1, 4], [4, 5], [1, 11]], [1, 2, 3], columns)


def test_user_precondition_error_message():
    """Test that user precondition errors show appropriate messages."""
    # Create test data - note that column 'x' does not exist
    df = _test_df_with_columns(["a", "b"])

    # Create pipeline with user precondition that will fail
    pline = pdp.PdPipeline(
        [pdp.FreqDrop(2, "a", prec=pdp.cond.HasAllColumns(["x"]))]
    )

    # Should raise FailedPreconditionError
    with pytest.raises(Exception) as exc_info:
        pline.apply(df)

    error_msg = str(exc_info.value).lower()

    # Check if error message indicates user precondition failure
    assert (
        "column x" in error_msg or "user-provided precondition" in error_msg
    ), f"Expected user precondition error message, got: {exc_info.value}"


def test_stage_precondition_error_message():
    """Test that stage precondition errors still work correctly."""
    # Create test data without column 'a' to trigger stage precondition failure
    df = _test_df_with_columns(["x", "y"])

    # Create pipeline without user precondition -
    # should trigger stage precondition
    pline = pdp.PdPipeline([pdp.FreqDrop(2, "a")])  # column 'a' missing

    # Should raise FailedPreconditionError
    with pytest.raises(Exception) as exc_info:
        pline.apply(df)

    error_msg = str(exc_info.value).lower()

    # Check if this shows stage precondition error
    assert (
        "column a" in error_msg and "freqdrop" in error_msg
    ), f"Expected stage precondition error message, got: {exc_info.value}"


def test_successful_pipeline_execution():
    """Test that normal operation still works with user preconditions."""
    # Create test data where everything should work
    df = _test_df_with_columns(["a", "b"])

    # Create pipeline with user precondition that should pass
    pline = pdp.PdPipeline(
        [pdp.FreqDrop(2, "a", prec=pdp.cond.HasAllColumns(["a", "b"]))]
    )

    # Should not raise any exceptions
    result = pline.apply(df)

    # Basic sanity checks
    assert result is not None
    assert isinstance(result, pd.DataFrame)


def test_user_vs_stage_precondition_distinction():
    """Test that user and stage precondition failures are distinguishable.

    This is the core test for issue #70 - ensuring that error messages
    properly indicate whether a user-provided condition failed vs a stage's
    built-in condition.

    """
    df = _test_df_with_columns(["a", "b"])

    # Test 1: User precondition failure (column 'x' doesn't exist)
    pline_user_fail = pdp.PdPipeline(
        [pdp.FreqDrop(2, "a", prec=pdp.cond.HasAllColumns(["x"]))]
    )

    with pytest.raises(Exception) as user_exc:
        pline_user_fail.apply(df)

    # Test 2: Stage precondition failure
    # (column 'z' doesn't exist for FreqDrop)
    df_no_z = _test_df_with_columns(["a", "b"])
    pline_stage_fail = pdp.PdPipeline([pdp.FreqDrop(2, "z")])

    with pytest.raises(Exception) as stage_exc:
        pline_stage_fail.apply(df_no_z)

    # The error messages should be different
    user_msg = str(user_exc.value).lower()
    stage_msg = str(stage_exc.value).lower()

    # User error should mention the missing column from user precondition
    assert (
        "column x" in user_msg or "user-provided precondition" in user_msg
    ), f"User error should mention column 'x', got: {user_exc.value}"

    # Stage error should mention the stage operation
    asrt_msg = "Stage error should mention column 'z' and FreqDrop, "
    asrt_msg += f"got: {stage_exc.value}"
    assert "column z" in stage_msg and "freqdrop" in stage_msg, asrt_msg
