"""Testing ColRename pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import SetIndex


def test_set_index():
    """Testing the SetIndex pipeline stage."""
    df = pd.DataFrame(
        data=[[8, 'a', 4], [5, 'b', 92]],
        index=[1, 2],
        columns=['num', 'char', 'w']
    )
    res_df = SetIndex('num', drop=True).apply(df)
    assert 'w' in res_df.columns
    assert 'num' not in res_df.columns
    assert 'char' in res_df.columns
    assert res_df.index.name == 'num'

    res_df = SetIndex(['w', 'num'], drop=True).apply(df)
    assert 'w' not in res_df.columns
    assert 'num' not in res_df.columns
    assert 'char' in res_df.columns
