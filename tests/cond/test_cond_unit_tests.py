"""Unit tests for Condition objects."""

import pytest
# import numpy as np
import pandas as pd
import pdpipe as pdp


NA_DF = pd.DataFrame(
    [[None, 1, 2], [None, None, 5]], [1, 2], ['ph', 'grade', 'age'])

NA_DF2 = pd.DataFrame(
    [[8, 1, 2], [1, 2, 5]], [1, 2], ['ph', 'grade', 'age'])


def test_basic_condition_stuff():
    # check expected behaviour of unfittable conditions
    cond = pdp.cond.HasNoMissingValues()
    assert not cond(NA_DF)
    # fittable should be False by default
    assert cond(NA_DF2)

    # check expected behaviour of fitted conditions
    cond = pdp.cond.HasNoMissingValues(fittable=True)
    cond.fit(NA_DF2)
    assert cond(NA_DF)
    assert not cond.fit_transform(NA_DF)
    assert not cond(NA_DF2)

    # check automatic call to fit_transform on __call__ of unfitted fittable
    cond = pdp.cond.HasNoMissingValues(fittable=True)
    assert cond(NA_DF2)
    assert cond(NA_DF)

    # check correct response for binary operators with unsupported args
    with pytest.raises(TypeError):
        cond & 5
    with pytest.raises(TypeError):
        cond ^ 5
    with pytest.raises(TypeError):
        cond | 5


def test_PerColumnCondition():
    conditions = [
        lambda x: x.isna().sum() > 0,
        lambda x: any([k == 5 for k in x.values]),
    ]
    cond = pdp.cond.PerColumnCondition(
        conditions=conditions,
        conditions_reduce='all',
    )
    assert not cond(NA_DF)

    cond = pdp.cond.PerColumnCondition(
        conditions=conditions,
        conditions_reduce='any',
    )
    print(cond)
    assert cond(NA_DF)

    with pytest.raises(ValueError):
        cond = pdp.cond.PerColumnCondition(
            conditions=conditions,
            conditions_reduce='bad_value',
        )

    with pytest.raises(ValueError):
        cond = pdp.cond.PerColumnCondition(
            conditions=conditions,
            columns_reduce='bad_value',
        )


def test_HasAtMostMissingValues():
    with pytest.raises(ValueError):
        pdp.cond.HasAtMostMissingValues('34')
