"""Testing basic pipeline stages."""

import pytest
import pandas as pd

from pdpipe.basic_stages import ConditionValidator
from pdpipe.cond import HasNoMissingValues, HasNoColumn
from pdpipe.exceptions import FailedConditionError


DF1 = pd.DataFrame([[1, 4], [4, None], [1, 11]], [1, 2, 3], ['a', 'b'])


def test_condition_validator_basic():
    stage = ConditionValidator(HasNoMissingValues())
    with pytest.raises(FailedConditionError):
        stage(DF1)


def test_condition_validator_ignore():
    stage = ConditionValidator(HasNoMissingValues(), errors='ignore')
    res_df = stage(DF1)
    assert 1 in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_condition_validator_multi_default():
    stage = ConditionValidator([HasNoMissingValues(), HasNoColumn('k')])
    with pytest.raises(FailedConditionError):
        stage(DF1)


def test_condition_validator_multi_any():
    stage = ConditionValidator(
        conditions=[HasNoMissingValues(), HasNoColumn('k')],
        reducer=any,
    )
    res_df = stage(DF1, verbose=True)
    assert 1 in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_condition_validator_bad_input():
    stage = ConditionValidator([5, HasNoColumn('k')])
    with pytest.raises(ValueError):
        stage(DF1)


def test_condition_validator_lambda_pass():
    stage = ConditionValidator(
        conditions=[lambda df: True, HasNoColumn('k')],
    )
    res_df = stage(DF1, verbose=True)
    assert 1 in res_df.index
    assert 2 in res_df.index
    assert 3 in res_df.index


def test_condition_validator_lambda_fail():
    stage = ConditionValidator([lambda df: False, HasNoColumn('k')])
    with pytest.raises(FailedConditionError):
        stage(DF1, verbose=True)

    # Used as documentation example
    with pytest.raises(FailedConditionError):
        ConditionValidator(lambda df: len(df.columns) == 5).apply(DF1)


def test_condition_validator_custom_func():
    def _foo(df: pd.DataFrame) -> bool:
        return False
    _foo.__doc__ = None
    stage = ConditionValidator([_foo, HasNoColumn('k')])
    with pytest.raises(FailedConditionError):
        stage(DF1, verbose=True)
