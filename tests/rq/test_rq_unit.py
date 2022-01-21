"""Testing rq"""

import pytest
import pandas as pd
from pdpipe import rq


#     a   b
# 1   1   4
# 2   4   5
# 3  18  11
def _df1():
    return pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ['a', 'b'])


def test_rq():
    df = _df1()
    res = rq.ColValGt('a', 4)(df)
    assert res.eq([False, False, True]).all()
    res = rq.ColValGe('a', 4)(df)
    assert res.eq([False, True, True]).all()
    res = rq.ColValLt('a', 4)(df)
    assert res.eq([True, False, False]).all()
    res = rq.ColValLe('a', 4)(df)
    assert res.eq([True, True, False]).all()
    res = rq.ColValEq('a', 4)(df)
    assert res.eq([False, True, False]).all()

    # unary operators
    res = (~ rq.ColValLe('a', 4))(df)
    assert res.eq([False, False, True]).all()

    # binary operators
    res = (rq.ColValLe('a', 1) | rq.ColValGe('b', 11))(df)
    assert res.eq([True, False, True]).all()
    res = (rq.ColValLe('a', 1) & rq.ColValGe('b', 11))(df)
    assert res.eq([False, False, False]).all()
    res = (rq.ColValLe('a', 4) | rq.ColValGe('b', 5))(df)
    assert res.eq([True, True, True]).all()
    res = (rq.ColValLe('a', 4) ^ rq.ColValGe('b', 5))(df)
    assert res.eq([True, False, True]).all()

    # doc-less func
    q = rq.RowQualifier(lambda df: df['a'] > 4)
    print(q)

    # errors
    q = rq.ColValGe('a', 4)
    with pytest.raises(TypeError):
        q & '4'
    with pytest.raises(TypeError):
        q | '4'
    with pytest.raises(TypeError):
        q ^ '4'
