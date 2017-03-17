"""Testing basic pipline stages."""

import pandas as pd

from pdpipe.sklearn_stages import Encode


def _some_df():
    return pd.DataFrame(
        data=[[3.2, "acd", "x1"], [7.2, "alk", "x2"], [12.1, "alk", "x3"]],
        index=[1, 2, 3],
        columns=["ph", "lbl", "name"]
    )


def test_encode():
    """Basic binning test."""
    df = _some_df()
    encode_stage = Encode()
    res_df = encode_stage(df)
    assert 'lbl' in res_df.columns
    assert res_df['lbl'][1] == 0
    assert res_df['lbl'][2] == 1
    assert res_df['lbl'][3] == 1
    assert res_df['name'][1] == 0
    assert res_df['name'][2] == 1
    assert res_df['name'][3] == 2


def test_encode_with_args():
    """Basic binning test."""
    df = _some_df()
    encode_stage = Encode("lbl", drop=False)
    res_df = encode_stage(df, verbose=True)
    assert 'lbl' in res_df.columns
    assert res_df['lbl_enc'][1] == 0
    assert res_df['lbl_enc'][2] == 1
    assert res_df['lbl_enc'][3] == 1


def test_encode_with_exclude():
    """Basic binning test."""
    df = _some_df()
    encode_stage = Encode("lbl", exclude_columns="name")
    res_df = encode_stage(df)
    assert 'lbl' in res_df.columns
    assert res_df['lbl'][1] == 0
    assert res_df['lbl'][2] == 1
    assert res_df['lbl'][3] == 1
