"""Test the TfidfVectorizeTokenLists pipeline stage."""

import pytest
import pandas as pd
import pdpipe as pdp


DF = pd.DataFrame(
    data=[
        [23, ['live', 'full', 'cats', 'mango']],
        [80, ['hovercraft', 'full', 'eels']],
    ],
    columns=['Age', 'Quote'],
)


DF2 = pd.DataFrame(
    data=[
        [29, ['eels', 'full', 'banana']],
    ],
    columns=['Age', 'Quote'],
)


@pytest.mark.parametrize("drop", [True, False])
def test_tfidf_vec(drop):
    tf = pdp.TfidfVectorizeTokenLists('Quote', drop=drop)
    res_df = tf(DF)
    assert 'Age' in res_df.columns
    words = []
    for i, row in DF.iterrows():
        words.extend(row['Quote'])
    uwords = set(words)
    for w in uwords:
        assert w in res_df.columns
    non_zeros = (
        res_df.drop('Quote', axis=1, errors='ignore') > 0
    ).T.sum().values
    for i, row in DF.iterrows():
        assert len(row['Quote']) == non_zeros[i] - 1
    if not drop:
        assert 'Quote' in res_df.columns

    res_df2 = tf(DF2)
    assert 'Age' in res_df2.columns
    if not drop:
        assert 'Quote' in res_df2.columns
    non_zeros2 = (
        res_df2.drop('Quote', axis=1, errors='ignore') > 0
    ).T.sum().values
    assert non_zeros2[0] - 1 == 2


def test_tfidf_vec_hierarchical_labels():
    tf = pdp.TfidfVectorizeTokenLists('Quote', hierarchical_labels=True)
    res_df = tf(DF)
    assert 'Age' in res_df.columns
    words = []
    for i, row in DF.iterrows():
        words.extend(row['Quote'])
    uwords = set(words)
    for w in uwords:
        assert f'Quote_{w}' in res_df.columns
    non_zeros = (
        res_df.drop('Quote', axis=1, errors='ignore') > 0
    ).T.sum().values
    for i, row in DF.iterrows():
        assert len(row['Quote']) == non_zeros[i] - 1
