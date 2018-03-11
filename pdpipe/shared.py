"""Shared inner functionalities for pdpipe."""


def _interpret_columns_param(columns):
    if isinstance(columns, str):
        return [columns]
    elif hasattr(columns, '__iter__'):
        return columns
    else:
        return [columns]


def _list_str(listi):
    if listi is None:
        return None
    if isinstance(listi, (list, tuple)):
        return ', '.join([str(elem) for elem in listi])
    return listi
