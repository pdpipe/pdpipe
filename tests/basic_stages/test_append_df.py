"""Testing basic pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import AppendDf


df1 = pd.DataFrame(
        data=[[3, 165,  'Dana'], [2, 172, 'Jane'], [4, 170, 'Nick']],
        columns=['Medals', 'Height', 'Name']
        )

df2 = pd.DataFrame(
        data=[[1, 163,  'Diana'], [5, 180, 'Bolt']],
        columns=['Medals', 'Height', 'Name']
        )


def test_append_basic():
    """Testing the append pipeline stage."""
    pd_result = df1.append(df2)
    pdp_result = AppendDf(df2).apply(df1)
    assert pdp_result.equals(pd_result)
