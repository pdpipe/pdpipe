"""Shared inner functionalities for pdpipe."""

import inspect
from typing import List, Iterable


def _interpret_columns_param(columns: object) -> List[object]:
    if isinstance(columns, str):
        return [columns]
    if hasattr(columns, '__iter__'):
        return columns
    return [columns]


def _list_str(listi: Iterable[object]) -> str:
    if listi is None:
        return None
    if isinstance(listi, (list, tuple)):
        return ', '.join([str(elem) for elem in listi])
    return listi


def _get_args_list(func: callable) -> List[str]:
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def _identity_function(x: object) -> object:
    return x


def _always_true(x: object) -> bool:
    """A function that always returns True."""
    return True
