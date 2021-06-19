"""Testing basic pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import MergeDf


df1 = pd.DataFrame(
        data=[[4, 165, 'Dana'], [2, 180, 'Jane'], [3, 170, 'Nick']],
        columns=['Medals', 'Height', 'Name']
        )

df2 = pd.DataFrame(
        data=[['F', 26, 'Jane'], ['M', 28, 'Nick'],  ['M', 27, 'Bolt']],
        columns=['Gender', 'Age', 'Name']
        )


def test_merge_inner():
    """Testing the default (inner) join pipeline stage."""
    pd_result = df1.merge(df2, on=['Name'], how='inner')
    pdp_result = MergeDf(df2, on=['Name'], how='inner').apply(df1)
    assert pdp_result.equals(pd_result)


def test_merge_left():
    """Testing the left join merge pipeline stage."""
    pd_result = df1.merge(df2, on=['Name'], how='left')
    pdp_result = MergeDf(df2, on=['Name'], how='left').apply(df1)
    assert pdp_result.equals(pd_result)


def test_merge_right():
    """Testing the right join merge pipeline stage."""
    pd_result = df1.merge(df2, on=['Name'], how='right')
    pdp_result = MergeDf(df2, on=['Name'], how='right').apply(df1)
    assert pdp_result.equals(pd_result)


def test_merge_outer():
    """Testing the outer join merge pipeline stage."""
    pd_result = df1.merge(df2, on=['Name'], how='outer')
    pdp_result = MergeDf(df2, on=['Name'], how='outer').apply(df1)
    assert pdp_result.equals(pd_result)
