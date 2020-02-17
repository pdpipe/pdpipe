"""Testing OneHotEncode pipeline stages."""

import pytest
import pandas as pd

from pdpipe.col_generation import OneHotEncode
from pdpipe import cq


def _one_categ_df():
    return pd.DataFrame([["USA"], ["UK"], ["Greece"]], [1, 2, 3], ["Born"])


def _one_categ_single_row_df():
    return pd.DataFrame([["Greece"]], [1], ["Born"])


def _one_categ_df_large():
    return pd.DataFrame(
        data=[["USA"], ["UK"], ["Greece"], ["USA"], ["USA"], ["UK"], ["UK"]],
        index=[1, 2, 3, 4, 5, 6, 7],
        columns=["Born"],
    )


@pytest.mark.onehotencode
def test_onehotencode_one():
    """Basic binning test."""
    df = _one_categ_df()
    onehotencode = OneHotEncode("Born")
    res_df = onehotencode(df, verbose=True)
    assert "Born" not in res_df.columns
    assert "Born_Greece" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_Greece" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0


@pytest.mark.onehotencode
def test_onehotencode_large():
    """Basic binning test."""
    df = _one_categ_df()
    onehotencode = OneHotEncode("Born")
    res_df = onehotencode(df, verbose=True)
    assert "Born" not in res_df.columns
    assert "Born_Greece" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _one_categ_df_large()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert len(res_df2) == 7
    assert "Born" not in res_df2.columns
    assert "Born_Greece" not in res_df2.columns
    assert res_df2["Born_UK"][3] == 0
    assert res_df2["Born_USA"][3] == 0
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][2] == 1
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 1


@pytest.mark.onehotencode
def test_onehotencode_no_drop_first():
    """Basic binning test."""
    df = _one_categ_df()
    onehotencode = OneHotEncode("Born", drop_first=False)
    res_df = onehotencode(df, verbose=True)
    assert "Born" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0
    assert "Born_Greece" in res_df.columns
    assert res_df["Born_Greece"][1] == 0
    assert res_df["Born_Greece"][2] == 0
    assert res_df["Born_Greece"][3] == 1

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
    assert "Born_Greece" in res_df2.columns
    assert res_df2["Born_Greece"][1] == 1


def _two_categ_df():
    return pd.DataFrame(
        data=[["USA", "Bob"], ["UK", "Jack"], ["Greece", "Yan"]],
        index=[1, 2, 3],
        columns=["Born", "Name"],
    )


def _two_categ_single_row_df():
    return pd.DataFrame([["Greece", "Bob"]], [1], ["Born", "Name"])


@pytest.mark.onehotencode
def test_onehotencode_two():
    """Basic binning test."""
    df = _two_categ_df()
    onehotencode = OneHotEncode()
    res_df = onehotencode(df)
    assert "Born" not in res_df.columns
    assert "Born_Greece" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0
    assert "Name" not in res_df.columns
    assert "Name_Bob" not in res_df.columns
    assert "Name_Jack" in res_df.columns
    assert res_df["Name_Jack"][1] == 0
    assert res_df["Name_Jack"][2] == 1
    assert res_df["Name_Jack"][3] == 0
    assert "Name_Yan" in res_df.columns
    assert res_df["Name_Yan"][1] == 0
    assert res_df["Name_Yan"][2] == 0
    assert res_df["Name_Yan"][3] == 1

    # check when fitted
    df2 = _two_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_Greece" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
    assert "Name" not in res_df.columns
    assert "Name_Bob" not in res_df.columns
    assert "Name_Jack" in res_df.columns
    assert res_df2["Name_Jack"][1] == 0
    assert "Name_Yan" in res_df.columns
    assert res_df2["Name_Yan"][1] == 0


@pytest.mark.onehotencode
def test_onehotencode_one_with_exclude():
    """Basic binning test."""
    df = _two_categ_df()
    onehotencode = OneHotEncode(exclude_columns=["Name"])
    print(onehotencode._col_arg)
    res_df = onehotencode(df)
    assert "Born" not in res_df.columns
    assert "Name" in res_df.columns
    assert "Name_Bob" not in res_df.columns
    assert "Name_Jack" not in res_df.columns
    assert "Name_Yan" not in res_df.columns
    assert "Greece" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _two_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_Greece" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
    assert "Name" in res_df.columns
    assert res_df2["Name"][1] == "Bob"
    assert "Name_Bob" not in res_df.columns
    assert "Name_Jack" not in res_df.columns
    assert "Name_Yan" not in res_df.columns


def _one_categ_df_with_nan():
    return pd.DataFrame([["USA"], ["UK"], [None]], [1, 2, 3], ["Born"])


@pytest.mark.onehotencode
def test_onehotencode_with_nan():
    """Basic binning test."""
    df = _one_categ_df_with_nan()
    onehotencode = OneHotEncode("Born")
    res_df = onehotencode(df)
    print(res_df)
    assert "Born" not in res_df.columns
    assert "Born_UK" not in res_df.columns
    assert "Born_nan" not in res_df.columns
    assert "Born_USA" in res_df.columns
    assert len(res_df.columns) == 1
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_UK" not in res_df2.columns
    assert "Born_nan" not in res_df2.columns
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0


@pytest.mark.onehotencode
def test_onehotencode_with_dummy_na():
    """Basic binning test."""
    df = _one_categ_df_with_nan()
    onehotencode = OneHotEncode("Born", dummy_na=True)
    res_df = onehotencode(df)
    assert "Born" not in res_df.columns
    assert "Born_nan" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_nan" not in res_df2.columns
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0


@pytest.mark.onehotencode
def test_onehotencode_with_dummy_na_no_drop_first():
    """Basic binning test."""
    df = _one_categ_df_with_nan()
    onehotencode = OneHotEncode("Born", dummy_na=True, drop_first=False)
    res_df = onehotencode(df)
    assert "Born" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0
    assert "Born_nan" in res_df.columns
    assert res_df["Born_nan"][1] == 0
    assert res_df["Born_nan"][2] == 0
    assert res_df["Born_nan"][3] == 1

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_Greece" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
    assert "Born_nan" in res_df.columns
    assert res_df2["Born_nan"][1] == 0


@pytest.mark.onehotencode
def test_onehotencode_one_no_drop():
    """Basic binning test."""
    df = _one_categ_df()
    onehotencode = OneHotEncode("Born", drop=False)
    res_df = onehotencode(df, verbose=True)
    assert "Greece" not in res_df.columns
    assert "Born" in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" in res_df2.columns
    assert res_df2["Born"][1] == "Greece"
    assert "Born_Greece" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0


@pytest.mark.onehotencode
def test_onehotencode_by_labels_cq():
    df = _two_categ_df()
    onehotencode = OneHotEncode(columns=cq.ByLabels(["Born", "Cat"]))
    res_df = onehotencode(df)
    assert "Born" not in res_df.columns
    assert "Born_Greece" not in res_df.columns
    assert "Born_UK" in res_df.columns
    assert res_df["Born_UK"][1] == 0
    assert res_df["Born_UK"][2] == 1
    assert res_df["Born_UK"][3] == 0
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0
    assert "Name" in res_df.columns
    assert "Name_Bob" not in res_df.columns
    assert "Name_Jack" not in res_df.columns
    assert "Name_Yan" not in res_df.columns

    # check when fitted
    df2 = _two_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_Greece" not in res_df2.columns
    assert "Born_UK" in res_df2.columns
    assert res_df2["Born_UK"][1] == 0
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
    assert "Name" in res_df.columns
    assert "Name_Bob" not in res_df.columns
    assert "Name_Jack" not in res_df.columns
    assert "Name_Yan" not in res_df.columns


@pytest.mark.onehotencode
@pytest.mark.parametrize("verbose", [True, False])
def test_onehotencode_one_with_drop_first_colname(verbose):
    df = _one_categ_df()
    onehotencode = OneHotEncode("Born", drop_first="UK")
    res_df = onehotencode(df, verbose=verbose)
    assert "Born" not in res_df.columns
    assert "Born_UK" not in res_df.columns
    assert "Born_Greece" in res_df.columns
    assert res_df["Born_Greece"][1] == 0
    assert res_df["Born_Greece"][2] == 0
    assert res_df["Born_Greece"][3] == 1
    assert "Born_USA" in res_df.columns
    assert res_df["Born_USA"][1] == 1
    assert res_df["Born_USA"][2] == 0
    assert res_df["Born_USA"][3] == 0

    # check when fitted
    df2 = _one_categ_single_row_df()
    assert onehotencode.is_fitted
    res_df2 = onehotencode(df2, verbose=True)
    print(res_df2)
    assert "Born" not in res_df2.columns
    assert "Born_UK" not in res_df2.columns
    assert "Born_Greece" in res_df2.columns
    assert res_df2["Born_Greece"][1] == 1
    assert "Born_USA" in res_df.columns
    assert res_df2["Born_USA"][1] == 0
