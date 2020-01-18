"""Unit test for column qualifiers."""

import pandas as pd
import pdpipe as pdp


NA_DF = pd.DataFrame(
    [[None, 1, 2], [None, None, 5]], [1, 2], ['ph', 'grade', 'age'])

NA_DF2 = pd.DataFrame(
    [[8, 1, 2], [1, 2, 5]], [1, 2], ['ph', 'grade', 'age'])


def test_with_at_most_missing_values():
    cq = pdp.cq.WithAtMostMissingValues(1)
    assert cq(NA_DF) == ['grade', 'age']
    assert cq(NA_DF2) == ['grade', 'age']


def test_unfittable_with_at_most_missing_values():
    cq = pdp.cq.WithAtMostMissingValues(1, fittable=False)
    assert cq(NA_DF) == ['grade', 'age']
    assert cq(NA_DF2) == ['ph', 'grade', 'age']


def test_not_operator():
    cq = ~ pdp.cq.WithAtMostMissingValues(1)
    assert cq(NA_DF) == ['ph']


NA_GLBL_DF = pd.DataFrame(
    [[None, 1, 2], [None, None, 5]], [1, 2], ['grep', 'grade', 'age'])


def test_and_operator():
    cq = pdp.cq.WithAtMostMissingValues(1) & pdp.cq.StartWith('gr')
    assert cq(NA_GLBL_DF) == ['grade']


NA_VARIOUS_FIRST_CHAR_DF = pd.DataFrame(
    [[1, 2, 3, 4], [5, 6, 7, None]], [1, 2], ['abe', 'bee', 'cry', 'no'])


def test_difference_operator():
    cq = pdp.cq.WithoutMissingValues() - pdp.cq.StartWith('b')
    assert cq(NA_VARIOUS_FIRST_CHAR_DF) == ['abe', 'cry']


MIXED_DTYPES_DF = pd.DataFrame(
    [['a', 2], ['b', 5]], [1, 2], ['char', 'init'])

# def test_
