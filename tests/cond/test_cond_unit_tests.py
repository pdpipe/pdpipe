"""Unit tests for Condition objects."""

import pytest
import numpy as np
import pandas as pd
import pdpipe as pdp


NA_DF = pd.DataFrame(
    [[None, 1, 2], [None, None, 5]], [1, 2], ['ph', 'grade', 'age'])

NA_DF2 = pd.DataFrame(
    [[8, 1, 2], [1, 2, 5]], [1, 2], ['ph', 'grade', 'age'])


def test_basic_condition_stuff():
    cond = pdp.cond.HasNoMissingValues()
    assert cond(NA_DF) is False
    # fittable should be False by default
    assert cond(NA_DF2) is True

    cond = pdp.cond.HasNoMissingValues(fittable=True)
    cond.fit(NA_DF2)
    assert cond(NA_DF) is True
    assert cond.fit_transform(NA_DF) is False
    assert cond(NA_DF2) is False
