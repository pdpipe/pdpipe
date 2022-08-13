import pandas as pd
import pdpipe as pdp


def _test_df():
    return pd.DataFrame([[2, 3],
                         [5, 6]],
                        columns=['a', 'b'])


def test_scaler():
    def scaling_decider(X: pd.DataFrame, y) -> str:
        """Determines with type of scaling to apply by examining all numerical columns."""
        for col in X.columns:
            if X[col].sum() % 3 == 0:
                return 'StandardScaler'
        return 'MinMaxScaler'

    scaler = pdp.Scale(
                # fit=False means it will take it from the application context, and not fit context
                scaler=pdp.dynamic(scaling_decider),
                joint=True,
        ),
    pipeline = pdp.PdPipeline(stages=[scaler])
    df = _test_df()
    res = pipeline.apply(df)
    expected_res = pd.DataFrame([[-1.264911, -0.632456],
                                 [0.632456, 1.264911]], columns=['a', 'b'])

    assert scaler.scaler == 'StandardScaler'
    pd.testing.assert_frame_equal(res, expected_res)
