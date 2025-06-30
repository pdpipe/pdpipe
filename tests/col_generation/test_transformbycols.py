"""Testing TransformByCols pipeline stages."""

import pytest
import pandas as pd
import pdpipe as pdp
from pdpipe.col_generation import TransformByCols

def ph_df():
    return pd.DataFrame(
        data=[[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]],
        index=[1, 2, 3],
        columns=["ph", "lbl"],
    )

def test_transformbycols_str():
    df = ph_df()
    cumsum_ph = TransformByCols("ph", "cumsum")
    res_df = cumsum_ph(df)
    assert res_df["ph"][1] == 3.2
    assert res_df["ph"][2] == 10.4
    assert res_df["ph"][3] == 22.5

def test_transformbycols_func():
    df = ph_df()
    def twice(s):
        return s * 2
    twice_ph = TransformByCols("ph", twice)
    res_df = twice_ph(df)
    assert res_df["ph"][1] == 6.4
    assert res_df["ph"][2] == 14.4
    assert res_df["ph"][3] == 24.2

def test_transformbycols_func_desc():
    df = ph_df()
    cumsum_ph = TransformByCols("ph", "cumsum", func_desc="Cumulative sum")
    res_df = cumsum_ph(df)
    assert res_df["ph"][1] == 3.2
    assert res_df["ph"][2] == 10.4
    assert res_df["ph"][3] == 22.5

def test_transformbycols_with_result_columns():
    df = ph_df()
    cumsum_ph = TransformByCols("ph", "cumsum", result_columns="ph_cumsum")
    res_df = cumsum_ph(df)
    assert "ph" not in res_df.columns
    assert res_df["ph_cumsum"][1] == 3.2
    assert res_df["ph_cumsum"][2] == 10.4
    assert res_df["ph_cumsum"][3] == 22.5

def test_transformbycols_with_drop():
    df = ph_df()
    cumsum_ph = TransformByCols("ph", "cumsum", drop=False)
    res_df = cumsum_ph(df)
    assert "ph" in res_df.columns
    assert "ph_trf" in res_df.columns
    assert res_df["ph_trf"][1] == 3.2
    assert res_df["ph_trf"][2] == 10.4
    assert res_df["ph_trf"][3] == 22.5

def test_transformbycols_no_drop_custom_suffix():
    df = ph_df()
    cumsum_ph = TransformByCols("ph", "cumsum", drop=False, suffix="_cumsum")
    res_df = cumsum_ph(df)
    assert "ph" in res_df.columns
    assert "ph_cumsum" in res_df.columns
    assert res_df["ph_cumsum"][1] == 3.2
    assert res_df["ph_cumsum"][2] == 10.4
    assert res_df["ph_cumsum"][3] == 22.5

def test_transformbycols_with_bad_len_result_columns():
    with pytest.raises(ValueError):
        TransformByCols("ph", "cumsum", result_columns=["a", "b"]) 