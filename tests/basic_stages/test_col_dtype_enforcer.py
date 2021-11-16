"""Test the ColumnDtypeEnforcer pipeline stage."""

import pandas as pd

from pdpipe import ColumnDtypeEnforcer
import pdpipe.cq as cq


DF = pd.DataFrame([[8, 'a'], [5, 'b']], [1, 2], ['num', 'initial'])


def test_dtype_enf_basic():
    stage = ColumnDtypeEnforcer({'num': float})
    res = stage(DF)
    assert res['num'].dtype == float


DF2 = pd.DataFrame(
    [[8, 2, 0], [5, 1, 1], [1, 2, 1]], [1, 2, 3], ['num1', 'num2', 'rank'])


def test_dtype_enf_col_qualifier():
    stage = ColumnDtypeEnforcer({'rank': bool, cq.StartWith('num'): float})
    res = stage(DF2)
    assert res['num1'].dtype == float
    assert res['num2'].dtype == float
    assert res['rank'].dtype == bool

    # Only col_qualifier as key, used as documentation example
    res = ColumnDtypeEnforcer({cq.StartWith('n'): float}).apply(DF)
    assert res['num'].dtype == float
