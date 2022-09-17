import pandas as pd
import pytest

import pdpipe as pdp
from pdpipe.exceptions import PipelineApplicationError


def _standard_scaler_df():
    return pd.DataFrame([[2, 3],
                         [5, 6]],
                        columns=['a', 'b'])


def _minmax_scaler_df():
    return pd.DataFrame([[2, 3],
                         [5, 7]],
                        columns=['a', 'b'])


def _scaling_decider_no_y(X: pd.DataFrame) -> str:
    """
    Determines with type of scaling to apply by examining all numerical
    columns.
    """
    for col in X.columns:
        print(X[col].sum())
        if X[col].sum() % 3 == 0:
            return 'StandardScaler'
    return 'MinMaxScaler'


def _scaling_decider_with_y(X: pd.DataFrame, y) -> str:
    """
    Determines with type of scaling to apply by examining all
    numerical columns.
    """
    if y.sum() % 2 == 0:
        return 'MinMaxScaler'
    return 'StandardScaler'


def _get_scaler(param_selector, fittable: bool = False):
    # fit=False, it will take it from the application context, not fit context
    return pdp.Scale(
        scaler=pdp.dynamic(param_selector, fittable=fittable),
        joint=True,
    )


def test_scaler_with_y():
    scaler = _get_scaler(_scaling_decider_with_y)
    pipeline = pdp.PdPipeline(stages=[scaler])
    df_standard_scaler = _standard_scaler_df()
    df_standard_scaler_x = df_standard_scaler['a'].to_frame()
    df_standard_scaler_y = df_standard_scaler['b']
    res_x, res_y = pipeline.fit_transform(df_standard_scaler_x,
                                          df_standard_scaler_y)
    expected_x = pd.DataFrame([-1, 1], columns=['a'])
    expected_y = pd.Series([3, 6])

    assert scaler.scaler == 'StandardScaler'
    pd.testing.assert_frame_equal(res_x, expected_x, check_dtype=False)
    pd.testing.assert_series_equal(res_y, expected_y)

    df_minmax_scaler = _minmax_scaler_df()
    df_minmax_scaler_x = df_minmax_scaler['a'].to_frame()
    df_minmax_scaler_y = df_minmax_scaler['b']
    res_x, res_y = pipeline.fit_transform(X=df_minmax_scaler_x,
                                          y=df_minmax_scaler_y)
    expected_x = pd.DataFrame([0, 1], columns=['a'])
    expected_y = pd.Series([3, 7])
    pd.testing.assert_series_equal(res_y, expected_y)

    assert scaler.scaler == 'MinMaxScaler'
    pd.testing.assert_frame_equal(res_x, expected_x, check_dtype=False)
    pd.testing.assert_series_equal(res_y, expected_y)


def test_scaler_no_y():
    scaler = _get_scaler(_scaling_decider_no_y)
    pipeline = pdp.PdPipeline(stages=[scaler])
    res = pipeline.apply(_standard_scaler_df())
    expected_res = pd.DataFrame([[-1.264911, -0.632456],
                                 [0.632456, 1.264911]], columns=['a', 'b'])

    assert scaler.scaler == 'StandardScaler'
    pd.testing.assert_frame_equal(res, expected_res)

    pipeline.fit(_minmax_scaler_df())
    res = pipeline.apply(_minmax_scaler_df())
    expected_res = pd.DataFrame([[0, 0.2],
                                 [0.6, 1]],
                                columns=['a', 'b'])

    assert scaler.scaler == 'MinMaxScaler'
    pd.testing.assert_frame_equal(res, expected_res)


def test_non_callable():
    with pytest.raises(PipelineApplicationError):
        scaler = _get_scaler('non_callable')
        pipeline = pdp.PdPipeline(stages=[scaler])
        pipeline.apply(_minmax_scaler_df())


def test_fittable():
    """ scaler should not change second time since the stage was fitted """
    scaler = _get_scaler(_scaling_decider_no_y, fittable=True)
    pipeline = pdp.PdPipeline(stages=[scaler])
    pipeline.apply(_minmax_scaler_df())
    assert scaler.scaler == 'MinMaxScaler'

    # check that minmax scaler was applied to standard scaler df
    res = pipeline.apply(_standard_scaler_df())
    expected_res = pd.DataFrame([[0, 0.2],
                                 [0.6, 0.8]],
                                columns=['a', 'b'])
    pd.testing.assert_frame_equal(res, expected_res)
