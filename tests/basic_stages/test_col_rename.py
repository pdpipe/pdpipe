"""Testing ColRename pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import ColRename


def test_colrename_with_dict():
    """Testing the ColRename pipeline stage."""
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


def test_colrename_with_func():
    """Testing the ColRename pipeline stage."""
    df = pd.DataFrame(
        data=[[8, 'a', 4], [5, 'b', 92]],
        index=[1, 2],
        columns=['num', 'char', 'w']
    )

    def renamer(lbl: str):
        if lbl.startswith('n'):
            return 'foo'
        return lbl
    res_df = ColRename(renamer).apply(df)
    assert 'w' in res_df.columns
    assert 'num' not in res_df.columns
    assert 'char' in res_df.columns
    assert 'foo' in res_df.columns


def test_colrename_with_documented_func():
    """Testing the ColRename pipeline stage."""
    df = pd.DataFrame(
        data=[[8, 'a', 4], [5, 'b', 92]],
        index=[1, 2],
        columns=['num', 'char', 'w']
    )

    def renamer(lbl: str):
        """Rename columns labeled 'n' to 'foo'."""
        if lbl.startswith('n'):
            return 'foo'
        return lbl
    res_df = ColRename(renamer).apply(df)
    assert 'w' in res_df.columns
    assert 'num' not in res_df.columns
    assert 'char' in res_df.columns
    assert 'foo' in res_df.columns
