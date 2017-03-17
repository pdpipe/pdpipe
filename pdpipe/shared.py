"""Shared inner functionalities for pdpipe."""


def _interpret_columns_param(columns, param_name):
    if isinstance(columns, str):
        return [columns]
    elif hasattr(columns, '__iter__'):
        if all(isinstance(arg, str) for arg in columns):
            return columns
        else:
            raise ValueError(
                "When {} is an iterable all its members should be "
                "strings.".format(param_name))
    else:
        raise ValueError(
            "Parameter {} should be either a string or an iterable of "
            "strings.".format(param_name))


def _list_str(listi):
    if listi is None:
        return None
    if isinstance(listi, (list, tuple)):
        return ', '.join([str(elem) for elem in listi])
    return listi
