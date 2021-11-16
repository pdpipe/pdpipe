"""Test PdPipelineAndSklearnEstimator."""

from typing import Optional

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import fbeta_score, make_scorer

import pdpipe as pdp
from pdpipe.skintegrate import (
    PdPipelineAndSklearnEstimator,
    pdpipe_scorer_from_sklearn_scorer,
)


DF1 = pd.DataFrame(
    data=[
        [23, 'Jo', 'M', True, 0.07, 'USA', 'Living life to its fullest'],
        [52, 'Regina', 'F', False, 0.26, 'Germany', 'I hate cats'],
        [23, 'Dana', 'F', True, 0.3, 'USA', 'the pen is mightier then the sword'],  # noqa : E501
        [25, 'Bo', 'M', False, 2.3, 'Greece', 'all for one and one for all'],
        [80, 'Richy', 'M', False, 100.2, 'Finland', 'I gots the dollarz'],
        [60, 'Paul', 'M', True, 1.87, 'Denmark', 'blah'],
        [44, 'Derek', 'M', True, 1.1, 'Denmark', 'every life is precious'],
        [72, 'Regina', 'F', True, 7.1, 'Greece', 'all of you get off my porch'],  # noqa : E501
        [50, 'Jim', 'M', False, 0.2, 'Germany', 'boy do I love dogs and cats'],
        [80, 'Wealthus', 'F', False, 123.2, 'Finland', 'me likey them moniez'],
    ],
    columns=['Age', 'Name', 'Gender', 'Smoking', 'Savings', 'Country', 'Quote'],  # noqa : E501
)


class MyPipelineAndModel(PdPipelineAndSklearnEstimator):

    def __init__(
        self,
        savings_max_val: Optional[int] = 100,
        drop_gender: Optional[bool] = False,
        scale_numeric: Optional[bool] = False,
        ohencode_country: Optional[bool] = True,
        savings_bin_val: Optional[int] = None,
        fit_intercept: Optional[bool] = True,
    ):
        self.savings_max_val = savings_max_val
        self.drop_gender = drop_gender
        self.scale_numeric = scale_numeric
        self.ohencode_country = ohencode_country
        self.savings_bin_val = savings_bin_val
        self.fit_intercept = fit_intercept
        cols_to_drop = []
        stages = [
            pdp.ColDrop(['Name', 'Quote'], errors='ignore'),
        ]
        if savings_bin_val:
            stages.append(pdp.Bin({'Savings': [savings_bin_val]}, drop=False))
            stages.append(pdp.Encode('Savings_bin'))
        if scale_numeric:
            stages.append(pdp.Scale('MinMaxScaler'))
        if drop_gender:
            cols_to_drop.append('Gender')
        else:
            stages.append(pdp.Encode('Gender'))
        if ohencode_country:
            stages.append(pdp.OneHotEncode('Country'))
        else:
            cols_to_drop.append('Country')
        stages.append(pdp.ColDrop(cols_to_drop, errors='ignore'))
        pline = pdp.PdPipeline(stages)
        model = LogisticRegression(fit_intercept=fit_intercept)
        super().__init__(pipeline=pline, estimator=model)


def test_pdpipeline_and_sklearn_model():
    mp = MyPipelineAndModel(
        savings_max_val=101,
        drop_gender=True,
        scale_numeric=True,
        ohencode_country=True,
        savings_bin_val=1,
        fit_intercept=True,
    )
    assert isinstance(mp.pipeline, pdp.PdPipeline)
    assert isinstance(mp.estimator, LogisticRegression)
    assert callable(mp.score)
    assert isinstance(str(mp), str)
    assert isinstance(repr(mp), str)

    # X-y subsets
    df = DF1.copy()
    x_lbls = ['Age', 'Gender', 'Savings', 'Country']
    all_x = df[x_lbls]
    all_y = df['Smoking']

    # check inheritence of predict, predict_proba, etc.
    mp.fit(all_x, all_y)
    res = mp.predict(all_x)
    assert isinstance(res, np.ndarray)
    assert res.dtype == bool
    res = mp.predict_proba(all_x)
    assert isinstance(res, np.ndarray)
    assert res.dtype == float
    res = mp.predict_log_proba(all_x)
    assert isinstance(res, np.ndarray)
    assert res.dtype == float
    res = mp.decision_function(all_x)
    assert isinstance(res, np.ndarray)
    assert res.dtype == float
    res = mp.score(all_x, all_y)
    assert isinstance(res, float)
    assert isinstance(mp.classes_, np.ndarray)
    assert mp.classes_.dtype == bool

    gcv = GridSearchCV(
        estimator=mp,
        param_grid={
            'savings_max_val': [99, 101],
            'scale_numeric': [True, False],
        },
        cv=3,
    )
    with pytest.raises(NotFittedError):
        check_is_fitted(gcv)
    gcv.fit(all_x, all_y)
    assert check_is_fitted(gcv) is None
    assert isinstance(gcv.cv_results_, dict)
    assert isinstance(gcv.best_estimator_, PdPipelineAndSklearnEstimator)
    score1 = gcv.best_score_

    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    my_scorer = pdpipe_scorer_from_sklearn_scorer(ftwo_scorer)
    assert isinstance(repr(my_scorer), str)
    gcv = GridSearchCV(
        estimator=mp,
        param_grid={
            'savings_max_val': [99, 101],
            'scale_numeric': [True, False],
        },
        cv=3,
        scoring=my_scorer,
    )
    gcv.fit(all_x, all_y)
    assert check_is_fitted(gcv) is None
    assert isinstance(gcv.cv_results_, dict)
    assert isinstance(gcv.best_estimator_, PdPipelineAndSklearnEstimator)
    score2 = gcv.best_score_
    assert score2 < score1


DF2 = pd.DataFrame(
    data=[['-1', 0], ['-1', 0], ['1', 1], ['1', 1]],
    index=[1, 2, 3, 4],
    columns=['feature1', 'target']
)


def test_pdpipeline_and_sklearn_model_documentation():
    all_x = DF2[['feature1']]
    all_y = DF2['target']
    mp = PdPipelineAndSklearnEstimator(
        pipeline=pdp.ColumnDtypeEnforcer({'feature1': int}),
        estimator=LogisticRegression()
    )
    mp.fit(all_x, all_y)
    res = mp.predict(all_x)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(DF2)
    assert res.dtype == all_y.dtype
