"""Shared inner functionalities for pdpipe."""

import inspect


def _interpret_columns_param(columns):
    if isinstance(columns, str):
        return [columns]
    if hasattr(columns, '__iter__'):
        return columns
    return [columns]


def _list_str(listi):
    if listi is None:
        return None
    if isinstance(listi, (list, tuple)):
        return ', '.join([str(elem) for elem in listi])
    return listi


def _get_args_list(func):
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def _identity_function(x):
    return x
