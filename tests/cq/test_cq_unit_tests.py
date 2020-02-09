"""Unit test for column qualifiers."""

import pytest
import numpy as np
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
    assert cq.fit_transform(NA_DF2) == ['ph', 'grade', 'age']
    # test fit
    assert cq.fit_transform(NA_DF) == ['grade', 'age']
    cq.fit(NA_DF2)
    assert cq(NA_DF) == ['ph', 'grade', 'age']


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


def test_xor_operator():
    cq = pdp.cq.WithAtMostMissingValues(1) ^ pdp.cq.StartWith('gr')
    assert cq(NA_GLBL_DF) == ['grep', 'age']


def test_or_operator():
    cq = pdp.cq.WithAtMostMissingValues(1) | pdp.cq.StartWith('gr')
    assert cq(NA_GLBL_DF) == ['grep', 'grade', 'age']


NA_VARIOUS_FIRST_CHAR_DF = pd.DataFrame(
    [[1, 2, 3, 4], [5, 6, 7, None]], [1, 2], ['abe', 'bee', 'cry', 'no'])


def test_difference_operator():
    cq = pdp.cq.WithoutMissingValues() - pdp.cq.StartWith('b')
    assert cq(NA_VARIOUS_FIRST_CHAR_DF) == ['abe', 'cry']


MIXED_DTYPES_DF = pd.DataFrame(
    [['ab', 2, 1.3], ['bc', 5, 2.2]], [1, 2], ['str', 'int', 'float'])


def test_by_column_condition():
    cq = pdp.cq.ByColumnCondition(lambda x: x.dtype == np.int64)
    assert cq(MIXED_DTYPES_DF) == ['int']


def test_of_dtype():
    print(MIXED_DTYPES_DF)
    print(MIXED_DTYPES_DF.dtypes)
    cq = pdp.cq.OfDtypes(np.number)
    assert cq(MIXED_DTYPES_DF) == ['int', 'float']
    cq = pdp.cq.OfDtypes([np.number, object])
    assert cq(MIXED_DTYPES_DF) == ['str', 'int', 'float']
    cq = pdp.cq.OfDtypes(np.int64)
    assert cq(MIXED_DTYPES_DF) == ['int']


def test_operator_attribute_errors():
    cq = pdp.cq.WithAtMostMissingValues(1)
    with pytest.raises(TypeError):
        cq & 'ad'
    with pytest.raises(TypeError):
        cq | 4
    with pytest.raises(TypeError):
        cq ^ (lambda x: ['a'])
    with pytest.raises(TypeError):
        cq - 32


MIXED_LABELS_DF = pd.DataFrame(
    [[8, 1, 2], [1, 2, 5]], [1, 2], ['ph', 'grad', 48])


def test_start_with():
    cq = pdp.cq.StartWith('g')
    assert cq(MIXED_LABELS_DF) == ['grad']


def test_columns_to_qualifier():
    cq = pdp.cq.StartWith('g')
    b = pdp.cq.columns_to_qualifier(cq)
    assert b == cq
