"""Testing the ColReorder stage."""

import pytest
import pandas as pd

from pdpipe import ColReorder
from pdpipe.exceptions import FailedPreconditionError


def _df():
    return pd.DataFrame([[8, 4, 3, 7]], columns=['a', 'b', 'c', 'd'])


def test_colreorder():
    df = _df()
    stage = ColReorder({'b': 0, 'c': 3})
    res = stage(df)
    assert list(res.columns) == ['b', 'a', 'd', 'c']

    stage2 = ColReorder({'g': 0})
    with pytest.raises(FailedPreconditionError):
        stage2(df)

    stage3 = ColReorder({'a': 8})
    with pytest.raises(ValueError):
        stage3(df)
