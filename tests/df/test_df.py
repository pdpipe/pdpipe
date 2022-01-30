"""Testing pdpipe's df module."""

import pandas as pd

import pdpipe as pdp
from pdpipe.df.df_transformer import _DataFrameMethodTransformer
from pdpipe import df


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


def test_df_set_axis():
    """Testing the ColDrop pipeline stage."""
    df = _test_df()  # noqa: F811
    assert 'num1' in df.columns
    stage = pdp.df.set_index(keys='num2')
    res_df = stage.apply(df)
    assert 'num1' in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns
    assert list(res_df.index) == [2, 4]

    pline = pdp.PdPipeline([
        pdp.df.set_index(keys='num2'),
        pdp.ColDrop('char'),
    ])
    stage = pline['set_index']
    assert isinstance(stage, _DataFrameMethodTransformer)
    res_df = pline(df)
    assert 'num1' in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' not in res_df.columns
    assert list(res_df.index) == [2, 4]

    # check inplace is ignored correctly
    df = _test_df()
    assert 'num1' in df.columns
    stage = pdp.df.set_index(keys='num2', drop=True, inplace=True)
    res_df = stage.apply(df)
    assert 'num1' in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns
    assert list(res_df.index) == [2, 4]
    assert 'num2' in df


def get_pipeline() -> pdp.PdPipeline:
    return pdp.PdPipeline([
        df.drop('char', axis=1),
        df.set_index(keys='num2'),
    ])


def test_more_df():
    df = _test_df()
    pline = get_pipeline()
    res_df = pline(df)
    assert 'num1' in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' not in res_df.columns
    assert list(res_df.index) == [2, 4]
