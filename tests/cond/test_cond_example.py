"""pytest file built from tests/cond/cond_example.md"""


def session_00001_line_2():
    r"""
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = ~ pdp.cond.HasAllColumns(['num', 'chr'])
    >>> cond(df)
    False
    >>> cond = ~ pdp.cond.HasAllColumns(['num','go'])
    >>> cond(df)
    True
    """


def session_00002_line_14():
    r"""
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8, None],[5, 2]], [1,2], ['foo', 'bar'])
    >>> col_cond = pdp.cond.HasAllColumns(['foo', 'bar'])
    >>> missing_cond = pdp.cond.HasNoMissingValues()
    >>> (col_cond | missing_cond)(df)
    True
    >>> (col_cond & missing_cond)(df)
    False
    >>> df = pd.DataFrame([[8, 9],[5, 2]], [1,2], ['foo', 'bar'])
    >>> (col_cond & missing_cond)(df)
    True
    """


def session_00003_line_28():
    r"""
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8, None],[5, 2]], [1,2], ['foo', 'bar'])
    >>> col_cond = pdp.cond.HasAllColumns(['foo', 'bar'])
    >>> missing_cond = pdp.cond.HasNoMissingValues()
    >>> (col_cond ^ missing_cond)(df)
    True
    >>> df = pd.DataFrame([[8, 9],[5, 2]], [1,2], ['foo', 'bar'])
    >>> (col_cond ^ missing_cond)(df)
    False
    """
