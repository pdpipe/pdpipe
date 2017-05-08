"""Testing ColRename pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import ColRename


def test_valdrop_with_columns():
    """Testing the ColDrop pipeline stage."""
    df = pd.DataFrame(
        data=[[8, 'a', 4], [5, 'b', 92]],
        index=[1, 2],
        columns=['num', 'char', 'w']
    )
    res_df = ColRename({'num': 'len', 'char': 'initial'}).apply(df)
    assert 'w' in res_df.columns
    assert 'num' not in res_df.columns
    assert 'char' not in res_df.columns
    assert 'len' in res_df.columns
    assert 'initial' in res_df.columns
