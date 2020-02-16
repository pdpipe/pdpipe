"""Testing basic pipeline stages."""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_approx_equal

from pdpipe import Log


def _some_df():
    return pd.DataFrame(
        data=[[1, 3.2, "acd"], [8, 7.2, "alk"], [3, 12.1, "alk"]],
        index=[1, 2, 3],
        columns=["rank", "ph", "lbl"]
    )


def _some_df2():
    return pd.DataFrame(
        data=[[3, 4.4, "acd"], [5, 6.1, "alk"], [1, 1.3, "alk"]],
        index=[1, 2, 3],
        columns=["rank", "ph", "lbl"]
    )


@pytest.mark.log
def test_log():
    df = _some_df()
    log_stage = Log()
    res_df = log_stage(df)
    assert 'rank' in res_df.columns
    assert 'ph' in res_df.columns
    for col in df.columns:
        for i in df.index:
            assert res_df[col][i] == df[col][i]
    assert res_df['rank_log'][1] == 0
    assert_approx_equal(res_df['rank_log'][2], 2.079441, significant=5)
    assert_approx_equal(res_df['rank_log'][3], 1.098612, significant=5)
    assert_approx_equal(res_df['ph_log'][1], 1.163151, significant=5)
    assert_approx_equal(res_df['ph_log'][2], 1.974081, significant=5)
    assert_approx_equal(res_df['ph_log'][3], 2.493205, significant=5)

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = log_stage(df2)
    assert 'rank' in res_df2.columns
    assert 'ph' in res_df2.columns
    for col in df2.columns:
        for i in df2.index:
            assert res_df2[col][i] == df2[col][i]
    assert_approx_equal(res_df2['rank_log'][1], 1.098612, significant=5)
    assert_approx_equal(res_df2['rank_log'][2], 1.609437, significant=5)
    assert res_df2['rank_log'][3] == 0
    assert_approx_equal(res_df2['ph_log'][1], 1.481604, significant=5)
    assert_approx_equal(res_df2['ph_log'][2], 1.808288, significant=5)
    assert_approx_equal(res_df2['ph_log'][3], 0.262364, significant=5)

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = log_stage.fit_transform(df2)
    assert 'rank' in res_df2.columns
    assert 'ph' in res_df2.columns
    for col in df2.columns:
        for i in df2.index:
            assert res_df2[col][i] == df2[col][i]
    assert_approx_equal(res_df2['rank_log'][1], 1.098612, significant=5)
    assert_approx_equal(res_df2['rank_log'][2], 1.609437, significant=5)
    assert res_df2['rank_log'][3] == 0
    assert_approx_equal(res_df2['ph_log'][1], 1.481604, significant=5)
    assert_approx_equal(res_df2['ph_log'][2], 1.808288, significant=5)
    assert_approx_equal(res_df2['ph_log'][3], 0.262364, significant=5)


def _non_neg_df():
    return pd.DataFrame(
        data=[[-2, 3.2, "acd"], [8, 7.2, "alk"], [3, 12.1, "alk"]],
        index=[1, 2, 3],
        columns=["rank", "ph", "lbl"]
    )


def _non_neg_df2():
    return pd.DataFrame(
        data=[[-3, 4.4, "acd"], [5, 6.1, "alk"], [1, 1.3, "alk"]],
        index=[1, 2, 3],
        columns=["rank", "ph", "lbl"]
    )


@pytest.mark.log
def test_log_non_neg():
    df = _non_neg_df()
    log_stage = Log(non_neg=True)
    res_df = log_stage(df)
    assert 'rank' in res_df.columns
    assert 'ph' in res_df.columns
    for col in df.columns:
        for i in df.index:
            assert res_df[col][i] == df[col][i]
    assert res_df['rank_log'][1] == -np.inf
    assert_approx_equal(res_df['rank_log'][2], 2.302585, significant=5)
    assert_approx_equal(res_df['rank_log'][3], 1.609436, significant=5)
    assert_approx_equal(res_df['ph_log'][1], 1.163151, significant=5)
    assert_approx_equal(res_df['ph_log'][2], 1.974081, significant=5)
    assert_approx_equal(res_df['ph_log'][3], 2.493205, significant=5)

    # see only transform (no fit) when already fitted
    df2 = _non_neg_df2()
    res_df2 = log_stage(df2, verbose=True)
    assert 'rank' in res_df2.columns
    assert 'ph' in res_df2.columns
    for col in df2.columns:
        for i in df2.index:
            assert res_df2[col][i] == df2[col][i]
    assert np.isnan(res_df2['rank_log'][1])
    assert_approx_equal(res_df2['rank_log'][2], 1.945910, significant=5)
    assert_approx_equal(res_df2['rank_log'][3], 1.098612, significant=5)
    assert_approx_equal(res_df2['ph_log'][1], 1.481604, significant=5)
    assert_approx_equal(res_df2['ph_log'][2], 1.808288, significant=5)
    assert_approx_equal(res_df2['ph_log'][3], 0.262364, significant=5)

    # check fit_transform when already fitted
    df2 = _some_df2()
    res_df2 = log_stage.fit_transform(df2)
    assert 'rank' in res_df2.columns
    assert 'ph' in res_df2.columns
    for col in df2.columns:
        for i in df2.index:
            assert res_df2[col][i] == df2[col][i]
    assert_approx_equal(res_df2['rank_log'][1], 1.098612, significant=5)
    assert_approx_equal(res_df2['rank_log'][2], 1.609437, significant=5)
    assert res_df2['rank_log'][3] == 0
    assert_approx_equal(res_df2['ph_log'][1], 1.481604, significant=5)
    assert_approx_equal(res_df2['ph_log'][2], 1.808288, significant=5)
    assert_approx_equal(res_df2['ph_log'][3], 0.262364, significant=5)


@pytest.mark.log
def test_log_non_neg_n_const_shift():
    df = _non_neg_df()
    log_stage = Log(non_neg=True, const_shift=0.1)
    res_df = log_stage(df)
    assert 'rank' in res_df.columns
    assert 'ph' in res_df.columns
    for col in df.columns:
        for i in df.index:
            assert res_df[col][i] == df[col][i]
    assert_approx_equal(res_df['rank_log'][1], -2.302585, significant=5)
    assert_approx_equal(res_df['rank_log'][2], 2.312534, significant=5)
    assert_approx_equal(res_df['rank_log'][3], 1.629240, significant=5)
    assert_approx_equal(res_df['ph_log'][1], 1.193922, significant=5)
    assert_approx_equal(res_df['ph_log'][2], 1.987874, significant=5)
    assert_approx_equal(res_df['ph_log'][3], 2.501435, significant=5)

    # see only transform (no fit) when already fitted
    df2 = _non_neg_df2()
    res_df2 = log_stage(df2, verbose=True)
    assert 'rank' in res_df2.columns
    assert 'ph' in res_df2.columns
    for col in df2.columns:
        for i in df2.index:
            assert res_df2[col][i] == df2[col][i]
    assert np.isnan(res_df2['rank_log'][1])
    assert_approx_equal(res_df2['rank_log'][2], 1.960094, significant=5)
    assert_approx_equal(res_df2['rank_log'][3], 1.131402, significant=5)
    assert_approx_equal(res_df2['ph_log'][1], 1.504077, significant=5)
    assert_approx_equal(res_df2['ph_log'][2], 1.824549, significant=5)
    assert_approx_equal(res_df2['ph_log'][3], 0.336472, significant=5)


# def test_encode_with_args():
#     """Basic binning test."""
#     df = _some_df()
#     encode_stage = Encode("lbl", drop=False)
#     res_df = encode_stage(df, verbose=True)
#     assert 'lbl' in res_df.columns
#     assert res_df['lbl_enc'][1] == 0
#     assert res_df['lbl_enc'][2] == 1
#     assert res_df['lbl_enc'][3] == 1
#
#     # see only transform (no fit) when already fitted
#     df2 = _some_df2()
#     res_df2 = encode_stage(df2)
#     assert 'lbl' in res_df.columns
#     assert res_df2['lbl_enc'][1] == 1
#     assert res_df2['lbl_enc'][2] == 0
#     assert res_df2['lbl_enc'][3] == 1
#
#     # check fit_transform when already fitted
#     df2 = _some_df2()
#     res_df2 = encode_stage.fit_transform(df2, verbose=True)
#     assert 'lbl' in res_df.columns
#     assert res_df2['lbl_enc'][1] == 1
#     assert res_df2['lbl_enc'][2] == 0
#     assert res_df2['lbl_enc'][3] == 1

@pytest.mark.log
def test_log_with_exclude():
    df = _some_df()
    log_stage = Log(exclude_columns='lbl')
    res_df = log_stage(df)
    assert 'lbl' in res_df.columns
    assert 'rank' in res_df.columns
    assert 'ph' in res_df.columns
    for col in df.columns:
        for i in df.index:
            assert res_df[col][i] == df[col][i]
    assert res_df['rank_log'][1] == 0
    assert_approx_equal(res_df['rank_log'][2], 2.079441, significant=5)
    assert_approx_equal(res_df['rank_log'][3], 1.098612, significant=5)
    assert_approx_equal(res_df['ph_log'][1], 1.163151, significant=5)
    assert_approx_equal(res_df['ph_log'][2], 1.974081, significant=5)
    assert_approx_equal(res_df['ph_log'][3], 2.493205, significant=5)


@pytest.mark.log
def test_log_with_verbose():
    df = _some_df()
    log_stage = Log()
    res_df = log_stage(df, verbose=True)
    assert 'lbl' in res_df.columns
    assert 'rank' in res_df.columns
    assert 'ph' in res_df.columns
    for col in df.columns:
        for i in df.index:
            assert res_df[col][i] == df[col][i]
    assert res_df['rank_log'][1] == 0
    assert_approx_equal(res_df['rank_log'][2], 2.079441, significant=5)
    assert_approx_equal(res_df['rank_log'][3], 1.098612, significant=5)
    assert_approx_equal(res_df['ph_log'][1], 1.163151, significant=5)
    assert_approx_equal(res_df['ph_log'][2], 1.974081, significant=5)
    assert_approx_equal(res_df['ph_log'][3], 2.493205, significant=5)


@pytest.mark.log
def test_log_drop():
    df = _some_df()
    log_stage = Log(drop=True)
    res_df = log_stage(df)
    assert 'rank' in res_df.columns
    assert 'ph' in res_df.columns
    assert 'rank_log' not in res_df.columns
    assert 'ph_log' not in res_df.columns
    assert res_df['rank'][1] == 0
    assert_approx_equal(res_df['rank'][2], 2.079441, significant=5)
    assert_approx_equal(res_df['rank'][3], 1.098612, significant=5)
    assert_approx_equal(res_df['ph'][1], 1.163151, significant=5)
    assert_approx_equal(res_df['ph'][2], 1.974081, significant=5)
    assert_approx_equal(res_df['ph'][3], 2.493205, significant=5)

    # see only transform (no fit) when already fitted
    df2 = _some_df2()
    res_df2 = log_stage(df2)
    assert 'rank' in res_df2.columns
    assert 'ph' in res_df2.columns
    assert 'rank_log' not in res_df2.columns
    assert 'ph_log' not in res_df2.columns
    assert_approx_equal(res_df2['rank'][1], 1.098612, significant=5)
    assert_approx_equal(res_df2['rank'][2], 1.609437, significant=5)
    assert res_df2['rank'][3] == 0
    assert_approx_equal(res_df2['ph'][1], 1.481604, significant=5)
    assert_approx_equal(res_df2['ph'][2], 1.808288, significant=5)
    assert_approx_equal(res_df2['ph'][3], 0.262364, significant=5)
