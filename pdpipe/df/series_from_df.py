"""Define callables that generate pandas.Series from pandas.DataFrame."""

import abc
from typing import Union, Tuple

import numpy
from pandas import DataFrame, Series


# === Series from DataFrame objects ======

class _SeriesFromDf(abc.ABC):
    """A serializable callable that returns a pandas.Series from input
    pandas.DataFrame objects."""

    @abc.abstractmethod
    def __call__(self, df: DataFrame) -> Series:
        raise NotImplementedError


try:
    _SeriesFromDfOperandType = Union[
        _SeriesFromDf,
        Series,
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
        numpy.complex128,
        numpy.str_,
        numpy.int64,
        numpy.uint64,
        numpy.float128,
        numpy.complex256,
        numpy.bool_,
        numpy.void,
        numpy.longlong,
        numpy.ulonglong,
        numpy.datetime64,
    ]
except AttributeError:  # pragma: no cover
    _SeriesFromDfOperandType = Union[
        _SeriesFromDf,
        Series,
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


class _SeriesFromDfByLabel(_SeriesFromDf):

    def __init__(self, column_label: object) -> None:
        self.column_label = column_label

    def __call__(self, df: DataFrame) -> Series:
        return df[self.column_label]

    def __repr__(self) -> str:
        return f"df[{self.column_label}]"


class _SeriesFromDfBySeriesMethod(_SeriesFromDf):

    def __init__(
        self,
        source: _SeriesFromDf,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        self.source = source
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        args_str = ""
        pos_args_str = ""
        if len(self.args) > 0:
            pos_args_str = ", ".join(map(str, self.args))
            args_str += pos_args_str
        kwargs_str = ""
        if len(self.kwargs) > 0:
            kwargs_str = ", ".join(
                f"{k}={v}" for k, v in self.kwargs.items())
            if len(self.args) > 0:
                args_str += ", "
            args_str += kwargs_str
        return (
            f"{self.source.__repr__()}.{self.method_name}({args_str})"
        )

    @staticmethod
    def _cast_series_from_df_to_series(
        args: tuple,
        kwargs: dict,
        df: DataFrame,
    ) -> Tuple[tuple, dict]:
        """Breaks down the input args tuple and kwargs dict,
        applies any _SeriesFromDf objects found to the input dataframe to

        Returns
        -------
        args : tuple
            The args tuple with any _SeriesFromDf objects replaced by the
            concrete Series objects.
        kwargs : Dict
            The kwargs dict with any _SeriesFromDf objects replaced by the
            concrete Series objects.
        """
        args_ = []
        for arg in args:
            if isinstance(arg, _SeriesFromDf):
                args_.append(arg(df))
            else:
                args_.append(arg)
        kwargs_ = {}
        for key, value in kwargs.items():
            if isinstance(value, _SeriesFromDf):
                kwargs_[key] = value(df)
            else:
                kwargs_[key] = value
        return tuple(args_), kwargs_

    def __call__(self, df: DataFrame) -> Series:
        source_res = self.source(df)
        method = getattr(source_res, self.method_name)
        args, kwargs = self._cast_series_from_df_to_series(
            self.args, self.kwargs, df)
        return method(*args, **kwargs)


# === repr help functions ======

def _arg_repr(arg: _SeriesFromDfOperandType) -> str:
    if isinstance(arg, Series):
        return "pandas.Series"
    return repr(arg)


# === Unary Operators ===

class _SeriesFromDfNeg(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
    ) -> None:
        self.first = first

    def __call__(self, df: DataFrame) -> Series:
        return - self.first(df)

    def __repr__(self) -> str:
        return f"-{_arg_repr(self.first)}"


# class _SeriesFromDfPos(_SeriesFromDf):
#
#     def __init__(
#         self,
#         first: _SeriesFromDf,
#     ) -> None:
#         self.first = first
#
#     def __call__(self, df: DataFrame) -> Series:
#         return + self.first(df)


class _SeriesFromDfAbs(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
    ) -> None:
        self.first = first

    def __call__(self, df: DataFrame) -> Series:
        return abs(self.first(df))

    def __repr__(self) -> str:
        return f"abs({_arg_repr(self.first)})"


class _SeriesFromDfInvert(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
    ) -> None:
        self.first = first

    def __call__(self, df: DataFrame) -> Series:
        return ~ self.first(df)

    def __repr__(self) -> str:
        return f"~{_arg_repr(self.first)}"


# === Binary Operators ===

# --- Boolean Operators ---

class _SeriesFromDfAnd(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) & self.second(df)
        except TypeError:
            return self.first(df) & self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} & {_arg_repr(self.second)}"


class _SeriesFromDfOr(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) | self.second(df)
            return self.first(df) | self.second
        if callable(self.second):
            return self.first | self.second(df)
        return self.first | self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} | {_arg_repr(self.second)}"


class _SeriesFromDfXor(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) ^ self.second(df)
            return self.first(df) ^ self.second
        if callable(self.second):
            return self.first ^ self.second(df)
        return self.first ^ self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} ^ {_arg_repr(self.second)}"


# --- Rich Comparison Operators ---

class _SeriesFromDfLt(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) < self.second(df)
        except TypeError:
            return self.first(df) < self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} < {_arg_repr(self.second)}"


class _SeriesFromDfLe(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) <= self.second(df)
        except TypeError:
            return self.first(df) <= self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} <= {_arg_repr(self.second)}"


class _SeriesFromDfEq(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) == self.second(df)
        except TypeError:
            return self.first(df) == self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} == {_arg_repr(self.second)}"


class _SeriesFromDfNe(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) != self.second(df)
        except TypeError:
            return self.first(df) != self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} != {_arg_repr(self.second)}"


class _SeriesFromDfGe(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) >= self.second(df)
        except TypeError:
            return self.first(df) >= self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} >= {_arg_repr(self.second)}"


class _SeriesFromDfGt(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) > self.second(df)
        except TypeError:
            return self.first(df) > self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} > {_arg_repr(self.second)}"


# --- Arithmetic Operators ---

class _SeriesFromDfAdd(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        try:
            return self.first(df) + self.second(df)
        except TypeError:
            return self.first(df) + self.second

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} + {_arg_repr(self.second)}"


class _SeriesFromDfSub(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) - self.second(df)
            return self.first(df) - self.second
        if callable(self.second):
            return self.first - self.second(df)
        return self.first - self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} - {_arg_repr(self.second)}"


class _SeriesFromDfMul(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) * self.second(df)
            return self.first(df) * self.second
        if callable(self.second):
            return self.first * self.second(df)
        return self.first * self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} * {_arg_repr(self.second)}"


class _SeriesFromDfTrueDiv(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) / self.second(df)
            return self.first(df) / self.second
        if callable(self.second):
            return self.first / self.second(df)
        return self.first / self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} / {_arg_repr(self.second)}"


class _SeriesFromDfFloorDiv(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) // self.second(df)
            return self.first(df) // self.second
        if callable(self.second):
            return self.first // self.second(df)
        return self.first // self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} // {_arg_repr(self.second)}"


class _SeriesFromDfMod(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) % self.second(df)
            return self.first(df) % self.second
        if callable(self.second):
            return self.first % self.second(df)
        return self.first % self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} % {_arg_repr(self.second)}"


class _SeriesFromDfPow(_SeriesFromDf):

    def __init__(
        self,
        first: _SeriesFromDf,
        second: _SeriesFromDfOperandType,
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, df: DataFrame) -> Series:
        if callable(self.first):
            if callable(self.second):
                return self.first(df) ** self.second(df)
            return self.first(df) ** self.second
        if callable(self.second):
            return self.first ** self.second(df)
        return self.first ** self.second  # pragma: no cover

    def __repr__(self) -> str:
        return f"{_arg_repr(self.first)} ** {_arg_repr(self.second)}"
