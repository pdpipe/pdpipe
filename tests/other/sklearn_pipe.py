"""Testing compatability with scikit-learn's Pipelinel objets."""

import pandas as pd
from pdutil.transform import x_y_by_col_lbl
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import pdpipe as pdp


def _train_df():
    return pd.DataFrame(
        data=[
            [3.2, "Pa", "A"],
            [7.2, "Pa", "B"],
            [12.1, "Ko", "B"],
            [18.6, "Pa", "A"],
            [5.3, "Pa", "A"],
            [5.0, "Ko", "A"],
            [17.9, "Ko", "B"],
            [1.3, "Ko", "C"],
        ],
        columns=["ph", "type", "lbl"]
    )


def _test_df():
    return pd.DataFrame(
        data=[
            [4.4, "Ko", "B"],
            [6.1, "Pa", "A"],
            [1.2, "Ko", "B"],
            [1.4, "Ko", "C"],
        ],
        columns=["ph", "lbl", "type"]
    )


def check_sk_pipeline():
    pline = pdp.make_pdpipeline(
        pdp.ApplyByCols("ph", lambda x: x - 1),
        # pdp.Bin({"ph": [0, 3, 5, 12]}),
        pdp.Encode(["type", "lbl"]),
    )
    print(pline)

    model_pline = make_pipeline(
        pdp.FreqDrop(2, "lbl"),
        LogisticRegression(),
    )
    print(model_pline)

    train = _train_df()
    res_train = pline(train)
    print(f"Processed train set: {res_train}")
    x_train, y_train = x_y_by_col_lbl(res_train, "lbl")
    model_pline = model_pline.fit(x_train, y_train)
    print(f"Fitted model pipeline: {model_pline}")

    test = _test_df()
    res_test = pline(test)
    print(f"Processed test set: {res_test}")
    x_test, y_test = x_y_by_col_lbl(res_test, "lbl")
    predictions = model_pline.predict(x_test)
    print(f"predictions: {predictions}")


if __name__ == "__main__":
    check_sk_pipeline()
