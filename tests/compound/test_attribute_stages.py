"""Testing attribute pipeline stages."""

import pandas as pd

import pdpipe as pdp
from pdpipe.core import PdPipeline
from pdpipe.basic_stages import ColDrop
from pdpipe.col_generation import Bin


def _some_df():
    return pd.DataFrame(
        data=[[-3, 'Will'], [4, 'Jan'], [5, 'Tasha'], [9, 'Data']],
        index=[1, 2, 3, 4],
        columns=['speed', 'name'])


def test_attribute_stage():
    """Testing attribute pipeline stages."""
    pipeline = pdp.ColDrop('name').Bin({'speed': [5]}, drop=True)
    assert isinstance(pipeline, PdPipeline)
    assert isinstance(pipeline[0], ColDrop)
    assert isinstance(pipeline[1], Bin)
    df = _some_df()
    res_df = pipeline(df)
    assert 'speed' in res_df.columns
    assert 'name' not in res_df.columns
