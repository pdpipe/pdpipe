"""Testing basic pipeline stages."""

import pandas as pd
import pytest

from pdpipe import DropDuplicates
from pdpipe.exceptions import FailedPreconditionError


def _test_df():
    return pd.DataFrame([[8, 1], [8, 2], [9, 2]], [1, 2, 3], ['a', 'b'])


def test_drop_duplicates():
    df = _test_df()
    stage = DropDuplicates('a')
    res = stage.apply(df)
    assert 1 in res.index
    assert 2 not in res.index
    assert 3 in res.index

    stage = DropDuplicates('b')
    res = stage.apply(df)
    assert 1 in res.index
    assert 2 in res.index
    assert 3 not in res.index

    stage = DropDuplicates(['a', 'b'])
    res = stage.apply(df)
    assert 1 in res.index
    assert 2 in res.index
    assert 3 in res.index

    stage = DropDuplicates()
    res = stage.apply(df, verbose=True)
    assert 1 in res.index
    assert 2 in res.index
    assert 3 in res.index

    stage = DropDuplicates('c')
    with pytest.raises(FailedPreconditionError):
        stage.apply(df)
