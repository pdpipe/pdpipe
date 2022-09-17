"""Custom types for pdpipe."""

from typing import Union, List, Callable

import numpy
import pandas


ColumnsParamType = Union[object, List[object], Callable]
ColumnLabelsType = Union[object, List[object]]

_series_operand_types = [
    pandas.Series,
    int,
    float,
    complex,
    int,
    bool,
    bytes,
    str,
    memoryview,
    numpy.int8,
    numpy.uint8,
    numpy.float16,
    numpy.timedelta64,
    numpy.object_,
    numpy.int16,
    numpy.uint16,
    numpy.float32,
    numpy.complex64,
    numpy.bytes_,
    numpy.int32,
    numpy.uint32,
    numpy.float64,
    numpy.str_,
    numpy.int64,
    numpy.uint64,
    numpy.bool_,
    numpy.void,
    numpy.longlong,
    numpy.ulonglong,
    numpy.datetime64,
]

try:
    _series_operand_types.extend([
        numpy.complex128,
        numpy.float128,
    ])
except AttributeError:  # pragma: no cover
    # were on a system with no 128-bit floats
    pass
try:
    _series_operand_types.append(numpy.complex256)
except AttributeError:  # pragma: no cover
    pass


SeriesOperandTypesTuple = tuple(_series_operand_types)
