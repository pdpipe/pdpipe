"""Testing the FitOnly wrapper pipeline stage."""

import pandas as pd
import pytest

from pdpipe import ColDrop, FitOnly
from pdpipe.exceptions import FailedPreconditionError


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


def test_fitonly_w_coldrop():
    df = _test_df()
    stage = FitOnly(ColDrop('num1'))
    res = stage(df)
    assert 'num1' not in res.columns
    assert 'num2' in res.columns
    assert 'char' in res.columns

    df = _test_df()
    res = stage(df)
    assert 'num1' in res.columns
    assert 'num2' in res.columns
    assert 'char' in res.columns

    df = _test_df()
    res = stage.fit_transform(df)
    assert 'num1' not in res.columns
    assert 'num2' in res.columns
    assert 'char' in res.columns

    df = _test_df()
    res = stage(df, verbose=True)
    assert 'num1' in res.columns
    assert 'num2' in res.columns
    assert 'char' in res.columns


def test_fitonly_w_coldrop_missing_col():
    df = _test_df()
    stage = FitOnly(ColDrop('num3'))
    with pytest.raises(FailedPreconditionError):
        stage.apply(df)
