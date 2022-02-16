"""Defines the _BoundColumnPotential class."""

from types import MethodType
from typing import Union, Set, Dict, Tuple, Optional

import numpy
from pandas import DataFrame, Series

from ..shared import _list_str
from ..core import PdPipelineStage
from ..types import SeriesOperandTypesTuple
from .func_lists import (
    SERIES_TRANSFORMS_BLACKLIST,
    SERIES_TRANSFORMS_WHITELIST,
)
from .series_from_df import (
    _SeriesFromDf,
    _SeriesFromDfOperandType,
    _SeriesFromDfByLabel,
    _SeriesFromDfBySeriesMethod,
    _SeriesFromDfNeg,
    _SeriesFromDfAbs,
    _SeriesFromDfInvert,
    _SeriesFromDfAnd,
    _SeriesFromDfOr,
    _SeriesFromDfXor,
    _SeriesFromDfEq,
    _SeriesFromDfNe,
    _SeriesFromDfLt,
    _SeriesFromDfLe,
    _SeriesFromDfGt,
    _SeriesFromDfGe,
    _SeriesFromDfAdd,
    _SeriesFromDfSub,
    _SeriesFromDfMul,
    _SeriesFromDfTrueDiv,
    _SeriesFromDfMod,
    _SeriesFromDfPow,
    _SeriesFromDfFloorDiv,
)


class SeriesFromDfAssigner(PdPipelineStage):
    """Assigns a series to a column to input dataframes.

    Parameters
    ----------
    assign_to_column : str
        The label of the column to assign the series to. If it already exists
        in the input dataframe, it will be overwritten. Otherwise, it will be
        added to it.
    series_from_df : _SeriesFromDf or Series
        The df-to-series callable with which will be produced on stage
        application, or a series to assign.
    required_columns : Set of objects
        The labels of all columns required in input dataframes in order to
        calculate the series to assign to the destination column.
    source_column_potential : _BoundColumnPotential, optional
        The _BoundColumnPotential that `series_from_df` came from. Optional.
    """

    _DOC_TEMPLATE = "Assign column {} with {}"

    def __init__(
        self,
        assign_to_column: str,
        series_from_df: _SeriesFromDfOperandType,
        required_columns: Set[object],
        source_column_potential: Optional['_BoundColumnPotential'] = None,
        **kwargs,
    ) -> None:
        self.assign_to_column = assign_to_column
        self.series_from_df = series_from_df
        self.required_columns = required_columns
        self.source_column_potential = source_column_potential
        self._columns_str = _list_str(required_columns)
        exmsg = (
            f"Not all required columns {self._columns_str} "
            f"found in input dataframe!"
        )
        desc = SeriesFromDfAssigner._DOC_TEMPLATE.format(
            self.assign_to_column, self.series_from_df)
        super_kwargs = {
            'exmsg': exmsg,
            'desc': desc,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df: DataFrame) -> bool:
        return set(self.required_columns).issubset(df.columns)

    def _transform(self, df: DataFrame, verbose=None) -> DataFrame:
        return df.assign(**{self.assign_to_column: self.series_from_df})

    # === Binary Operators ===

    # --- Boolean Operators ---

    def __and__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential & other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df & other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfAnd(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfAnd(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __or__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential | other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # other.series_from_df | self.series_from_df should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfOr(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfOr(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __xor__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential ^ other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df ^ other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfXor(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfXor(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    # --- Rich Comparison Operators ---

    def __lt__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential < other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df < other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfGt(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfLt(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __le__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential <= other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df <= other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfGe(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfLe(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __eq__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential == other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df == other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfEq(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfEq(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented  # pragma: no cover

    def __ne__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential != other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df != other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfNe(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfNe(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented  # pragma: no cover

    def __ge__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential >= other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df >= other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfLe(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfGe(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __gt__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential > other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df > other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfLt(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfGt(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    # --- Arithmetic Operators ---

    def __add__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential + other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df + other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfAdd(
                    other.series_from_df, self.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfAdd(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __radd__(self, other: object) -> 'SeriesFromDfAssigner':
        return self.__add__(other)  # pragma: no cover

    def __sub__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential - other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df - other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfSub(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfSub(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __mul__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential * other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df * other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfMul(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfMul(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __truediv__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential / other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df / other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfTrueDiv(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfTrueDiv(
                    self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __floordiv__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential // other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df // other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfFloorDiv(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfFloorDiv(
                    self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __mod__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential % other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df % other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfMod(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfMod(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __pow__(self, other: object) -> 'SeriesFromDfAssigner':
        if isinstance(other, _BoundColumnPotential):
            if self.source_column_potential:
                new_col_potential = self.source_column_potential ** other
                return SeriesFromDfAssigner(
                    assign_to_column=self.assign_to_column,
                    series_from_df=new_col_potential.series_from_df,
                    required_columns=new_col_potential.required_columns,
                    source_column_potential=new_col_potential,
                )
            # else, self.series_from_df is pandas.Series or scalar so
            # self.series_from_df ** other should work
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfPow(
                    self.series_from_df, other.series_from_df),
                required_columns=self.required_columns,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            return SeriesFromDfAssigner(
                assign_to_column=self.assign_to_column,
                series_from_df=_SeriesFromDfPow(self.series_from_df, other),
                required_columns=self.required_columns,
            )
        return NotImplemented


try:
    _BoundColumnPotentialOperandType = Union[
        '_BoundColumnPotential',
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
    _BoundColumnPotentialOperandType = Union[
        '_BoundColumnPotential',
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


class _BoundColumnPotential():

    def __init__(
        self,
        series_from_df: _SeriesFromDf,
        required_columns: Set[object],
    ) -> None:
        self.series_from_df = series_from_df
        self.required_columns = required_columns

    # === Unary operators ===

    def __neg__(self) -> '_BoundColumnPotential':
        return _BoundColumnPotential(
            series_from_df=_SeriesFromDfNeg(self.series_from_df),
            required_columns=self.required_columns,
        )

    def __pos__(self) -> '_BoundColumnPotential':
        return self

    def __abs__(self) -> '_BoundColumnPotential':
        return _BoundColumnPotential(
            series_from_df=_SeriesFromDfAbs(self.series_from_df),
            required_columns=self.required_columns,
        )

    def __invert__(self) -> '_BoundColumnPotential':
        return _BoundColumnPotential(
            series_from_df=_SeriesFromDfInvert(self.series_from_df),
            required_columns=self.required_columns,
        )

    # === Binary operators ===

    # --- Boolean Operators ---

    def __and__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfAnd(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfAnd(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rand__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self & other  # pragma: no cover

    def __or__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfOr(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfOr(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __ror__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self | other  # pragma: no cover

    def __xor__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfXor(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfXor(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rxor__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self ^ other  # pragma: no cover

    # --- Rich Comparison Operators ---

    def __lt__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfLt(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfLt(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __le__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfLe(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfLe(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __eq__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfEq(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfEq(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented  # pragma: no cover

    def __ne__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfNe(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfNe(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented  # pragma: no cover

    def __ge__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfGe(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfGe(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __gt__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfGt(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfGt(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    # --- Arithmetic Operators ---

    def __add__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfAdd(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfAdd(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __radd__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__add__(other)

    def __sub__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfSub(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfSub(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rsub__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__sub__(other)

    def __mul__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfMul(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfMul(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rmul__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__mul__(other)

    def __truediv__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfTrueDiv(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfTrueDiv(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rtruediv__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__truediv__(other)

    def __floordiv__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfFloorDiv(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfFloorDiv(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rfloordiv__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__floordiv__(other)

    def __mod__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfMod(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfMod(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rmod__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__mod__(other)

    def __pow__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        if isinstance(other, _BoundColumnPotential):
            series_from_df = _SeriesFromDfPow(
                first=self.series_from_df,
                second=other.series_from_df,
            )
            req_cols = self.required_columns | other.required_columns
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=req_cols,
            )
        if isinstance(other, SeriesOperandTypesTuple):
            series_from_df = _SeriesFromDfPow(
                first=self.series_from_df,
                second=other,
            )
            return _BoundColumnPotential(
                series_from_df=series_from_df,
                required_columns=self.required_columns,
            )
        return NotImplemented

    def __rpow__(
        self,
        other: _BoundColumnPotentialOperandType,
    ) -> '_BoundColumnPotential':
        return self.__pow__(other)

    # --- Our custom assignment operator ---

    def __lshift__(
        self,
        other: Union['_BoundColumnPotential', Series],
    ) -> PdPipelineStage:
        if isinstance(self.series_from_df, _SeriesFromDfByLabel):
            me: _SeriesFromDfByLabel = self.series_from_df
            try:
                return SeriesFromDfAssigner(
                    assign_to_column=me.column_label,
                    series_from_df=other.series_from_df,
                    required_columns=other.required_columns,
                    source_column_potential=other,
                )
            except AttributeError:  # other is Series
                return SeriesFromDfAssigner(
                    assign_to_column=me.column_label,
                    series_from_df=other,
                    required_columns=set(),
                )
        return NotImplemented  # pragma: no cover

    def __rshift__(
        self,
        other: Union['_BoundColumnPotential', Series],
    ) -> PdPipelineStage:
        if isinstance(other, _BoundColumnPotential):
            return other.__lshift__(self)
        # if isinstance(self.series_from_df, _SeriesFromDfByLabel):
        #     me: _SeriesFromDfByLabel = self.series_from_df
        #     try:
        #         return SeriesFromDfAssigner(
        #             assign_to_column=me.column_label,
        #             series_from_df=other.series_from_df,
        #             required_columns=other.required_columns,
        #             source_column_potential=other,
        #         )
        #     except AttributeError:  # other is Series
        #         return SeriesFromDfAssigner(
        #             assign_to_column=me.column_label,
        #             series_from_df=other,
        #             required_columns=set(),
        #         )
        return NotImplemented


# --- add series transformer methods to _BoundColumnPotential ---

class _BoundColumnPotentialSeriesMethodTransformerHandle():

    def __init__(self, method_name: str, doc: str) -> None:
        self._method_name = method_name
        self.__doc__ = doc

    @staticmethod
    def _cast_bound_col_potentials_to_series_from_df_objects(
        args: tuple, kwargs: dict,
    ) -> Tuple[Tuple, Dict, Set]:
        """Breaks down the input args tuple and kwargs dict,
        cast any included _BoundColumnPotential objects into _SeriesFromDf
        objects, and rebuilds the args tuple and kwargs dict, also constructing
        the the required column set, and returning all threeself.

        Returns
        -------
        args : tuple
            The args tuple with any _BoundColumnPotential objects replaced by
            _SeriesFromDf objects.
        kwargs : Dict
            The kwargs dict with any _BoundColumnPotential objects replaced by
            _SeriesFromDf objects.
        required_columns : set
            The set of all column labels required by the function.
        """
        required_columns = set()
        args_ = []
        for arg in args:
            if isinstance(arg, _BoundColumnPotential):
                args_.append(arg.series_from_df)
                required_columns |= arg.required_columns
            else:
                args_.append(arg)
        args = tuple(args_)
        kwargs_ = {}
        for k, v in kwargs.items():
            if isinstance(v, _BoundColumnPotential):
                kwargs_[k] = v.series_from_df
                required_columns |= v.required_columns
            else:
                kwargs_[k] = v
        kwargs = kwargs_
        return args, kwargs, required_columns

    def __call__(
        self,
        bound_col_instance: _BoundColumnPotential,
        *args,
        **kwargs: Dict[str, object],
    ) -> PdPipelineStage:
        res = self._cast_bound_col_potentials_to_series_from_df_objects(
            args=args,
            kwargs=kwargs,
        )
        args, kwargs, required_columns = res
        new_ser_from_df = _SeriesFromDfBySeriesMethod(
            source=bound_col_instance.series_from_df,
            method_name=self._method_name,
            args=args,
            kwargs=kwargs,
        )
        required_columns |= bound_col_instance.required_columns
        return _BoundColumnPotential(
            series_from_df=new_ser_from_df,
            required_columns=required_columns,
        )

    def __get__(self, instance, owner):
        return MethodType(self, instance) if instance else self


__RETURNS = 'Returns'
__SERIES = 'Series'


def _has_series_transform_doc(attr_name: str, attr: object) -> bool:
    if attr_name.startswith('_') or not callable(attr):
        return False
    try:
        doc_lines = attr.__doc__.split('\n')
        returns_line_index = None
        for i, line in enumerate(doc_lines):
            if __RETURNS in line:
                returns_line_index = i + 2
                break
        if returns_line_index:
            return_type_line = doc_lines[returns_line_index]
            if __SERIES in return_type_line:
                return True  # pragma: no cover
        return False
    except (AttributeError, IndexError):  # pragma: no cover
        return False


for attr_name in dir(Series):
    attr = getattr(Series, attr_name)
    add = False
    if attr_name in SERIES_TRANSFORMS_WHITELIST:
        add = True
    elif attr_name in SERIES_TRANSFORMS_BLACKLIST:
        add = False
    else:
        # check if documented return value points at a series-to-series func
        add = _has_series_transform_doc(attr_name, attr)
    if add:
        handle = _BoundColumnPotentialSeriesMethodTransformerHandle(
            method_name=attr_name,
            doc=attr.__doc__,
        )
        setattr(
            _BoundColumnPotential,
            attr_name,
            handle,
        )


# ==== factory method ====

def get_bound_column_potential_by_label(
    column_label: str,
) -> _BoundColumnPotential:
    """Return a bound column potential for a column with the given labelself.

    Parameters
    ----------
    column_label : str
            The label of the column to bind.

    Returns
    -------
    _BoundColumnPotential
            A bound column potential for the column with the given label.
    """
    return _BoundColumnPotential(
        series_from_df=_SeriesFromDfByLabel(column_label=column_label),
        required_columns={column_label},
    )
