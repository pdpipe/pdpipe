"""Testing pdp.fly."""

import pytest
import pandas as pd
from pdpipe.fly import (
    drop_rows_where,
    keep_rows_where,
)


#     a   b
# 1   1   4
# 2   4   5
# 3  18  11
def _df1():
    return pd.DataFrame([[1, 4], [4, 5], [18, 11]], [1, 2, 3], ['a', 'b'])


#        a   b
# 1      1   4
# 2   Null   5
# 3     18  11
def _df2():
    return pd.DataFrame([[1, 4], [None, 5], [18, 11]], [1, 2, 3], ['a', 'b'])


def test_drop_where():
    df = _df1()
    res = (drop_rows_where['a'] > 4)(df)
    assert (res.index == [1, 2]).all()
    res = (drop_rows_where['a'] >= 4)(df)
    assert (res.index == [1]).all()
    res = (drop_rows_where['a'] < 4)(df)
    assert (res.index == [2, 3]).all()
    res = (drop_rows_where['a'] <= 4)(df)
    assert (res.index == [3]).all()
    res = (drop_rows_where['a'] == 4)(df)
    assert (res.index == [1, 3]).all()
    res = (drop_rows_where['a'] != 4)(df)
    assert (res.index == [2]).all()
    res = (drop_rows_where['a'].isin([1, 3, 5, 18]))(df)
    assert (res.index == [2]).all()
    df2 = _df2()
    res = (drop_rows_where['a'].isna())(df2)
    assert (res.index == [1, 3]).all()
    res = (drop_rows_where['a'].notna())(df2)
    assert (res.index == [2]).all()

    # unary operators
    res = (~(drop_rows_where['a'] > 4))(df)
    assert (res.index == [3]).all()

    # binary operators
    res = ((drop_rows_where['a'] <= 1) | (drop_rows_where['b'] >= 11))(df)
    assert (res.index == [2]).all()
    res = ((drop_rows_where['a'] <= 1) & (drop_rows_where['b'] >= 11))(df)
    assert (res.index == [1, 2, 3]).all()
    res = ((drop_rows_where['a'] <= 4) ^ (drop_rows_where['b'] >= 5))(df)
    assert (res.index == [2]).all()

    # errors
    dstage = drop_rows_where['a'] > 4
    with pytest.raises(TypeError):
        dstage & '4'
    with pytest.raises(TypeError):
        dstage | '4'
    with pytest.raises(TypeError):
        dstage ^ '4'
    with pytest.raises(ValueError):
        drop_rows_where['x':'y'] == 4

    # verbose
    stage = drop_rows_where['a'] > 4
    res = stage.apply(df, verbose=True)
    assert (res.index == [1, 2]).all()


def test_keep_where():
    df = _df1()
    res = (keep_rows_where['a'] > 4)(df)
    assert (res.index == [3]).all()
    res = (keep_rows_where['a'] >= 4)(df)
    assert (res.index == [2, 3]).all()
    res = (keep_rows_where['a'] < 4)(df)
    assert (res.index == [1]).all()
    res = (keep_rows_where['a'] <= 4)(df)
    assert (res.index == [1, 2]).all()
    res = (keep_rows_where['a'] == 4)(df)
    assert (res.index == [2]).all()
    res = (keep_rows_where['a'] != 4)(df)
    assert (res.index == [1, 3]).all()
    res = (keep_rows_where['a'].isin([1, 3, 5, 18]))(df)
    assert (res.index == [1, 3]).all()
    df2 = _df2()
    res = (keep_rows_where['a'].isna())(df2)
    assert (res.index == [2]).all()
    res = (keep_rows_where['a'].notna())(df2)
    assert (res.index == [1, 3]).all()

    # unary operators
    res = (~(keep_rows_where['a'] > 4))(df)
    assert (res.index == [1, 2]).all()

    # binary operators
    res = ((keep_rows_where['a'] <= 1) | (keep_rows_where['b'] >= 11))(df)
    assert (res.index == [1, 3]).all()
    res = ((keep_rows_where['a'] <= 1) & (keep_rows_where['b'] >= 11))(df)
    assert (res.index == []).all()
    res = ((keep_rows_where['a'] <= 4) ^ (keep_rows_where['b'] >= 5))(df)
    assert (res.index == [1, 3]).all()

    # errors
    dstage = keep_rows_where['a'] > 4
    with pytest.raises(TypeError):
        dstage & '4'
    with pytest.raises(TypeError):
        dstage | '4'
    with pytest.raises(TypeError):
        dstage ^ '4'
    with pytest.raises(ValueError):
        keep_rows_where['x':'y'] == 4

    # verbose
    stage = keep_rows_where['a'] > 4
    res = stage.apply(df, verbose=True)
    assert (res.index == [3]).all()
