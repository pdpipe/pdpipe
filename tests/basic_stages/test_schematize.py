"""Testing the ColReorder stage."""

import pytest
import pandas as pd
import pdpipe as pdp

from pdpipe import Schematize
from pdpipe.exceptions import FailedPreconditionError


def _df():
    return pd.DataFrame([[2, 4, 8], [3, 6, 9]], [1, 2], ['a', 'b', 'c'])


def _df2():
    return pd.DataFrame([[2, 4], [3, 6]], [1, 2], ['a', 'b'])


def _df3():
    return pd.DataFrame([[2, 4], [3, 6]], [1, 2], ['a', 'c'])


def _df4():
    return pd.DataFrame([[2, 4, 8], [3, 6, 9]], [1, 2], ['b', 'c', 'a'])


def test_schematize():
    df = _df()
    stage = Schematize(['a', 'c'])
    res = stage(df)
    assert list(res.columns) == ['a', 'c']
    assert res.iloc[0, 0] == 2
    assert res.iloc[1, 0] == 3
    assert res.iloc[0, 1] == 8
    assert res.iloc[1, 1] == 9

    stage = Schematize(['c', 'b'])
    res = stage(df)
    assert list(res.columns) == ['c', 'b']
    assert res.iloc[0, 0] == 8
    assert res.iloc[1, 0] == 9
    assert res.iloc[0, 1] == 4
    assert res.iloc[1, 1] == 6

    stage = Schematize(['a', 'g'])
    with pytest.raises(FailedPreconditionError):
        stage(df)


@pytest.mark.xfail
def test_schematize_with_cq():
    # fit the stage on an [a, b] df
    df = _df2()
    stage = Schematize(pdp.cq.AllColumns())
    res = stage(df)
    assert list(res.columns) == ['a', 'b']

    # pass an [a, b, c] df to the fitted stage, make sure c is dropped
    df = _df()
    res = stage(df)
    assert list(res.columns) == ['a', 'b']

    # pass an [b, c, a] df to the fitted stage, make sure c is dropped and
    # a, b are correctly ordered
    df = _df4()
    res = stage(df)
    assert list(res.columns) == ['a', 'b']

    # check that a df with [a, c] columns fails a stage fitted with [a, b]
    with pytest.raises(FailedPreconditionError):
        stage(_df3())
