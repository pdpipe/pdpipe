"""Testing basic pipeline stages."""

import pandas as pd
import pytest

from pdpipe.basic_stages import AppendDf
from pdpipe.exceptions import FailedPreconditionError


def _get_test_df1():
    return pd.DataFrame(data=[[3, 165, 'Dana'],
                              [2, 172, 'Jane'],
                              [4, 170, 'Nick']],
                        columns=['Medals', 'Height', 'Name']
                        )


def _get_test_df2():
    return pd.DataFrame(data=[[1, 163, 'Diana'], [5, 180, 'Bolt']],
                        columns=['Medals', 'Height', 'Name']
                        )


def test_append_basic():
    """Testing the append pipeline stage."""

    df1 = _get_test_df1()
    df2 = _get_test_df2()
    pd_result = df1.append(df2)
    pdp_result = AppendDf(df2).apply(df1)
    assert pdp_result.equals(pd_result)


def test_append_error():
    """Testing the append pipeline stage.
    - df2 columns are not subset of df1"""

    df1 = _get_test_df1()
    df2 = _get_test_df2()
    df2 = df2.rename(columns={'Medals': 'medals'})
    stage = AppendDf(df2)
    with pytest.raises(FailedPreconditionError):
        stage.fit(df1, verbose=True)
