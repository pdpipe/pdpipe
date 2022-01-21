"""Row qualifiers for pdpipe."""

from typing import List, Set, Union
from numbers import Number

import pandas


class RowQualifier(object):
    """An object that returns a boolean series from input dataframes.

    The boolean series will be of the length of the input dataframe, and so can
    be used as a boolean index to get a subset of the input dataframe's rows.

    Parameters
    ----------
    func : callable
        The function to apply to input dataframes.
    """

    def __init__(self, func: callable) -> None:
        self._rqfunc = func

    def __call__(self, df: pandas.DataFrame) -> pandas.Series:
        return self._rqfunc(df)

    def __repr__(self):
        if self._rqfunc.__doc__:  # pragma: no cover
            return f"<RowQualifier: Qualify rows with {self._rqfunc.__doc__}>"
        return "<RowQualifier: Qualify rows by function>"

    # --- overriding boolean operators ---

    class _AndQualifierFunc(object):
        """A pickle-able AND qualifier callable."""

        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __call__(self, df):
            return self.first(df) & self.second(df)

    def __and__(self, other):
        try:
            res_func = RowQualifier._AndQualifierFunc(
                first=self._rqfunc,
                second=other._rqfunc,
            )
            res_func.__doc__ = (
                f"{self._rqfunc.__doc__ or 'Anonymous qualifier 1'} AND "
                f"{other._rqfunc.__doc__ or 'Anonymous qualifier 2'}"
            )
            return RowQualifier(func=res_func)
        except AttributeError:
            return NotImplemented

    class _OrQualifierFunc(object):
        """A pickle-able OR qualifier callable."""

        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __call__(self, df):
            return self.first(df) | self.second(df)

    def __or__(self, other):
        try:
            res_func = RowQualifier._OrQualifierFunc(
                first=self._rqfunc,
                second=other._rqfunc,
            )
            res_func.__doc__ = (
                f"{self._rqfunc.__doc__ or 'Anonymous qualifier 1'} OR "
                f"{other._rqfunc.__doc__ or 'Anonymous qualifier 2'}"
            )
            return RowQualifier(func=res_func)
        except AttributeError:
            return NotImplemented

    class _XorQualifierFunc(object):
        """A pickle-able OR qualifier callable."""

        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __call__(self, df):
            return self.first(df) ^ self.second(df)

    def __xor__(self, other):
        try:
            res_func = RowQualifier._XorQualifierFunc(
                first=self._rqfunc,
                second=other._rqfunc,
            )
            res_func.__doc__ = (
                f"{self._rqfunc.__doc__ or 'Anonymous qualifier 1'} XOR "
                f"{other._rqfunc.__doc__ or 'Anonymous qualifier 2'}"
            )
            return RowQualifier(func=res_func)
        except AttributeError:
            return NotImplemented

    class _NotQualifierFunc(object):
        """A pickle-able NOT row qualifier callable."""

        def __init__(self, rq):
            self.rq = rq

        def __call__(self, df):
            return ~ self.rq(df)

    def __invert__(self):
        res_func = RowQualifier._NotQualifierFunc(rq=self._rqfunc)
        res_func.__doc__ = (
            f"NOT {self._rqfunc.__doc__ or 'Anonymous qualifier'}"
        )
        return RowQualifier(func=res_func)


class ColValGt(RowQualifier):
    """A row qualifier that qualifies rows with a value greater than a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value : Number
        The value to check against.
    """

    class _GtRowFunc(object):
        """A pickle-able gt callable class."""

        def __init__(self, label: object, value: Number) -> None:
            self.label = label
            self.value = value
            self.__doc__ = f"df[{label}] > {value}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].gt(self.value)

    def __init__(self, label: object, value: Number) -> None:
        super().__init__(
            func=ColValGt._GtRowFunc(label, value),
        )


class ColValGe(RowQualifier):
    """A row qualifier that qualifies rows with a value greater than or equal
    to a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value : Number
        The value to check against.
    """

    class _GeRowFunc(object):
        """A pickle-able ge callable class."""

        def __init__(self, label: object, value: Number) -> None:
            self.label = label
            self.value = value
            self.__doc__ = f"df[{label}] => {value}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].ge(self.value)

    def __init__(self, label: object, value: Number) -> None:
        super().__init__(
            func=ColValGe._GeRowFunc(label, value),
        )


class ColValLt(RowQualifier):
    """A row qualifier that qualifies rows with a value lesser than a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value : Number
        The value to check against.
    """

    class _LtRowFunc(object):
        """A pickle-able lt callable class."""

        def __init__(self, label: object, value: Number) -> None:
            self.label = label
            self.value = value
            self.__doc__ = f"df[{label}] < {value}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].lt(self.value)

    def __init__(self, label: object, value: Number) -> None:
        super().__init__(
            func=ColValLt._LtRowFunc(label, value),
        )


class ColValLe(RowQualifier):
    """A row qualifier that qualifies rows with a value lesser than or equal
    to a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value : Number
        The value to check against.
    """

    class _LeRowFunc(object):
        """A pickle-able le callable class."""

        def __init__(self, label: object, value: Number) -> None:
            self.label = label
            self.value = value
            self.__doc__ = f"df[{label}] <= {value}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].le(self.value)

    def __init__(self, label: object, value: Number) -> None:
        super().__init__(
            func=ColValLe._LeRowFunc(label, value),
        )


class ColValEq(RowQualifier):
    """A row qualifier that qualifies rows with a value equal to a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value : Number
        The value to check against.
    """

    class _EqRowFunc(object):
        """A pickle-able eq callable class."""

        def __init__(self, label: object, value: Number) -> None:
            self.label = label
            self.value = value
            self.__doc__ = f"df[{label}] == {value}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].eq(self.value)

    def __init__(self, label: object, value: Number) -> None:
        super().__init__(
            func=ColValEq._EqRowFunc(label, value),
        )


class ColValNe(RowQualifier):
    """A row qualifier that qualifies rows with a value not equal to a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value : Number
        The value to check against.
    """

    class _NeRowFunc(object):
        """A pickle-able ne callable class."""

        def __init__(self, label: object, value: Number) -> None:
            self.label = label
            self.value = value
            self.__doc__ = f"df[{label}] != {value}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].ne(self.value)

    def __init__(self, label: object, value: Number) -> None:
        super().__init__(
            func=ColValNe._NeRowFunc(label, value),
        )


class ColValIsIn(RowQualifier):
    """A row qualifier that qualifies rows with a value not equal to a value.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    value_list : list of object
        The list of values to check against.
    """

    class _IsInRowFunc(object):
        """A pickle-able isin callable class."""

        def __init__(
            self,
            label: object,
            value_list: Union[List[object], Set[object]],
        ) -> None:
            self.label = label
            self.value_list = value_list
            self.__doc__ = f"df[{label}] is in {value_list}"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].isin(self.value_list)

    def __init__(
        self,
        label: object,
        value_list: Union[List[object], Set[object]],
    ) -> None:
        super().__init__(
            func=ColValIsIn._IsInRowFunc(label, value_list),
        )


class ColValIsNa(RowQualifier):
    """A row qualifier that qualifies rows with a null value in a column.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    """

    class _IsNaRowFunc(object):
        """A pickle-able isna callable class."""

        def __init__(
            self,
            label: object,
        ) -> None:
            self.label = label
            self.__doc__ = f"df[{label}] is NA"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].isna()

    def __init__(
        self,
        label: object,
    ) -> None:
        super().__init__(func=ColValIsNa._IsNaRowFunc(label))


class ColValNotNa(RowQualifier):
    """A row qualifier that qualifies rows with a non-NA value in a column.

    Parameters
    ----------
    label : object
        The label of the column the qualifier checks.
    """

    class _NotNaRowFunc(object):
        """A pickle-able notna callable class."""

        def __init__(
            self,
            label: object,
        ) -> None:
            self.label = label
            self.__doc__ = f"df[{label}] is not NA"

        def __call__(self, df: pandas.DataFrame) -> pandas.Series:
            return df[self.label].notna()

    def __init__(
        self,
        label: object,
    ) -> None:
        super().__init__(func=ColValNotNa._NotNaRowFunc(label))


del pandas
del Number
