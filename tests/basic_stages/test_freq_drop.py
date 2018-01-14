"""Testing basic pipeline stages."""

import pandas as pd

from pdpipe.basic_stages import FreqDrop


DF1 = pd.DataFrame([[1, 4], [4, 5], [1, 11]], [1, 2, 3], ['a', 'b'])


def test_freqdrop_basic():
    """Testing the FreqDrop pipeline stage."""
    res_df = FreqDrop(2, 'a').apply(DF1)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index


def test_freqdrop_verbose():
    """Testing the FreqDrop pipeline stage."""
    res_df = FreqDrop(2, 'a').apply(DF1, verbose=True)
    assert 1 in res_df.index
    assert 2 not in res_df.index
    assert 3 in res_df.index
