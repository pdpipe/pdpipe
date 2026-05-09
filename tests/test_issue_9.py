"""Regression test for pdpipe issue #9 (Diff transformer).

This is the BEFORE/AFTER probe pinned to the issue:
- BEFORE this PR (origin/master): `pdp.Diff` does not exist; `import
  pdp.Diff` raises AttributeError, demonstrating the gap.
- AFTER this PR: `pdp.Diff('val')` produces a column-wise first
  difference and composes with other stages.

Lives at the top level of `tests/` (matches the convention used for
test_issue_70.py — see PR #147).

"""

import numpy as np
import pandas as pd

import pdpipe as pdp


def test_issue_9_diff_stage_exists():
    # The original issue (2019-10-13) asks for a "Scikit like transformer"
    # for first-differences. This test fails on origin/master because
    # pdp.Diff is not defined.
    assert hasattr(
        pdp, "Diff"
    ), "pdpipe issue #9: pdp.Diff column-generation stage is missing"


def test_issue_9_basic_first_difference_matches_pandas():
    df = pd.DataFrame(
        data=[[100], [110], [95], [130]],
        columns=["val"],
    )
    res = pdp.Diff("val", drop=True)(df)
    expected = df["val"].diff()
    # diff with drop=True replaces the column in place; the resulting
    # series must be identical to pandas.Series.diff().
    pd.testing.assert_series_equal(
        res["val"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_issue_9_composes_in_pipeline():
    # The added value over `df.diff()` is being a pdpipe stage, so the
    # composability check is the load-bearing assertion.
    df = pd.DataFrame(
        data=[[1, 100], [2, 110], [3, 95], [4, 130]],
        columns=["t", "val"],
    )
    pipeline = pdp.PdPipeline(
        stages=[
            pdp.Diff("val"),
            pdp.ColRename({"val_diff": "val_delta"}),
        ]
    )
    res = pipeline(df)
    assert "val" in res.columns
    assert "val_delta" in res.columns
    assert np.isnan(res["val_delta"].iloc[0])
    assert res["val_delta"].iloc[1] == 10
