"""Testing basic pipeline stages."""

import pytest
import pandas as pd

import pdpipe as pdp
from pdpipe.sklearn_stages import Encode
from pdpipe.exceptions import UnfittedPipelineStageError


def _some_df():
    return pd.DataFrame(
        data=[[3.2, "acd", "x1"], [7.2, "alk", "x2"], [12.1, "alk", "x3"]],
        index=[1, 2, 3],
        columns=["ph", "lbl", "name"]
    )


def _some_df2():
    return pd.DataFrame(
        data=[[4.4, "alk", "x1"], [6.1, "acd", "x2"], [1.2, "alk", "x3"]],
        index=[1, 2, 3],
        columns=["ph", "lbl", "name"]
    )


def test_encode():
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

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage(df2)
    assert 'lbl' in res_df2.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1
    assert res_df2['name'][1] == 0
    assert res_df2['name'][2] == 1
    assert res_df2['name'][3] == 2

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage.transform(df2)
    assert 'lbl' in res_df2.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1
    assert res_df2['name'][1] == 0
    assert res_df2['name'][2] == 1
    assert res_df2['name'][3] == 2

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage.fit_transform(df2)
    assert 'lbl' in res_df.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1
    assert res_df2['name'][1] == 0
    assert res_df2['name'][2] == 1
    assert res_df2['name'][3] == 2


def test_encode_fit():
    df = _some_df()
    encode_stage = Encode()

    with pytest.raises(UnfittedPipelineStageError):
        encode_stage.transform(df)

    res_df = encode_stage.fit(df)
    assert 'lbl' in res_df.columns
    assert res_df['lbl'][1] == 'acd'
    assert res_df['lbl'][2] == 'alk'
    assert res_df['lbl'][3] == 'alk'

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage.transform(df2)
    assert 'lbl' in res_df2.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1
    assert res_df2['name'][1] == 0
    assert res_df2['name'][2] == 1
    assert res_df2['name'][3] == 2

    # check that apply only transforms (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage.transform(df2)
    assert 'lbl' in res_df2.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1
    assert res_df2['name'][1] == 0
    assert res_df2['name'][2] == 1
    assert res_df2['name'][3] == 2


def test_encode_with_args():
    df = _some_df()
    encode_stage = Encode("lbl", drop=False)
    res_df = encode_stage(df, verbose=True)
    assert 'lbl' in res_df.columns
    assert res_df['lbl_enc'][1] == 0
    assert res_df['lbl_enc'][2] == 1
    assert res_df['lbl_enc'][3] == 1

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage(df2)
    assert 'lbl' in res_df.columns
    assert res_df2['lbl_enc'][1] == 1
    assert res_df2['lbl_enc'][2] == 0
    assert res_df2['lbl_enc'][3] == 1

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = encode_stage.fit_transform(df2, verbose=True)
    assert 'lbl' in res_df.columns
    assert res_df2['lbl_enc'][1] == 1
    assert res_df2['lbl_enc'][2] == 0
    assert res_df2['lbl_enc'][3] == 1


def test_encode_with_exclude():
    df = _some_df()
    encode_stage = Encode("lbl", exclude_columns="name")
    res_df = encode_stage(df)
    assert 'lbl' in res_df.columns
    assert res_df['lbl'][1] == 0
    assert res_df['lbl'][2] == 1
    assert res_df['lbl'][3] == 1


def test_encode_in_pipeline():
    drop_name = pdp.ColDrop('name')
    encode_stage = Encode()
    pline = drop_name + encode_stage

    df = _some_df()
    res_df = pline(df)
    assert 'lbl' in res_df.columns
    assert 'name' not in res_df.columns
    assert res_df['lbl'][1] == 0
    assert res_df['lbl'][2] == 1
    assert res_df['lbl'][3] == 1

    # check fitted pipeline
    df2 = _some_df2()
    res_df2 = pline(df2)
    assert 'lbl' in res_df2.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1


def test_encode_in_pipelin_fit_n_transform():
    drop_name = pdp.ColDrop('name')
    encode_stage = Encode()
    pline = drop_name + encode_stage

    df = _some_df()

    with pytest.raises(UnfittedPipelineStageError):
        res_df = pline.transform(df)

    res_df = pline.fit(df)
    assert 'lbl' in res_df.columns
    assert 'name' in res_df.columns
    assert res_df['lbl'][1] == 'acd'
    assert res_df['lbl'][2] == 'alk'
    assert res_df['lbl'][3] == 'alk'

    res_df = pline.transform(df)
    assert 'lbl' in res_df.columns
    assert 'name' not in res_df.columns
    assert res_df['lbl'][1] == 0
    assert res_df['lbl'][2] == 1
    assert res_df['lbl'][3] == 1

    # check fitted pipeline
    df2 = _some_df2()
    res_df2 = pline.transform(df2)
    assert 'lbl' in res_df2.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = pline.fit_transform(df2, verbose=True)
    assert 'lbl' in res_df.columns
    assert 'name' not in res_df.columns
    assert res_df2['lbl'][1] == 1
    assert res_df2['lbl'][2] == 0
    assert res_df2['lbl'][3] == 1
