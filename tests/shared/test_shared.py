"""Testing shared functions of pdpipe."""

import numpy as np

from pdpipe.shared import (
    _interpret_columns_param,
    _list_str,
    _get_args_list,
    _identity_function,
    _str2list
)


def test_interpret_columns_param():
    assert _interpret_columns_param('a') == ['a']
    assert _interpret_columns_param(5) == [5]
    assert _interpret_columns_param([1, 2]) == [1, 2]


def test_list_str():
    assert _list_str(None) is None
    assert _list_str(['a', 'b']) == 'a, b'
    assert _list_str('a') == 'a'
    assert _list_str((1, 2)) == '1, 2'
    assert _list_str(5) == 5


def foo(a, b):
    return a + b


def test_get_args_list():
    print(_get_args_list(foo))
    assert _get_args_list(foo) == ['a', 'b']


def test_identity_function():
    assert _identity_function(5) == 5


def test_str2list():
    assert _str2list('a') == ['a']
    assert _str2list(['a']) == ['a']
    assert isinstance(_str2list(tuple('a')), tuple)
    assert _str2list(np.array(['a'])) == np.array(['a'])
