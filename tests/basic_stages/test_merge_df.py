"""Testing basic pipeline stages."""

import pandas as pd
import pytest

from pdpipe.basic_stages import MergeDf
from pdpipe.exceptions import FailedPreconditionError


def _get_test_df1():
    return pd.DataFrame(data=[[4, 165, 'Dana'],
                              [2, 180, 'Jane'],
                              [3, 170, 'Nick']],
                        columns=['Medals', 'Height', 'Name']
                        )


def _get_test_df2():
    return pd.DataFrame(data=[['F', 26, 'Jane'],
                              ['M', 28, 'Nick'],
                              ['M', 27, 'Bolt']],
                        columns=['Gender', 'Age', 'Name']
                        )


def test_merge_joins():
    """Testing the different joins of merge pipeline stage."""

    df1 = _get_test_df1()
    df2 = _get_test_df2()

    # Testing the default (inner) join pipeline stage.
    pd_result = df1.merge(df2, on=['Name'], how='inner')
    pdp_result = MergeDf(df2, on=['Name'], how='inner').apply(df1)
    assert pdp_result.equals(pd_result)

    # Testing the left join merge pipeline stage.
    pd_result = df1.merge(df2, on=['Name'], how='left')
    pdp_result = MergeDf(df2, on=['Name'], how='left').apply(df1)
    assert pdp_result.equals(pd_result)

    # Testing the right join pipeline stage.
    pd_result = df1.merge(df2, on=['Name'], how='right')
    pdp_result = MergeDf(df2, on=['Name'], how='right').apply(df1)
    assert pdp_result.equals(pd_result)

    # Testing the outer join pipeline stage.
    pd_result = df1.merge(df2, on=['Name'], how='outer')
    pdp_result = MergeDf(df2, on=['Name'], how='outer').apply(df1)
    assert pdp_result.equals(pd_result)


def test_merge_different_columns():
    """Testing the outer join merge pipeline stage."""

    df1 = _get_test_df1()
    df2 = _get_test_df2()

    df1 = df1.rename(columns={'Name': 'name'})

    stage = MergeDf(df2, on=['Name'], how='inner')
    with pytest.raises(FailedPreconditionError):
        stage.fit(df1, verbose=True)

    pd_result = df1.merge(df2, left_on=['name'], right_on=['Name'])
    pdp_result = MergeDf(df2, left_on=['name'],
                         right_on=['Name']).apply(df1)
    assert pdp_result.equals(pd_result)
