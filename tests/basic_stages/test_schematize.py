"""Testing the ColReorder stage."""

import pytest
import pandas as pd

from pdpipe import Schematize
from pdpipe.exceptions import FailedPreconditionError


def _df():
    return pd.DataFrame([[2, 4, 8], [3, 6, 9]], [1, 2], ['a', 'b', 'c'])


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
