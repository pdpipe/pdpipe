"""Basic pdpipe PdPipelineStages."""

from typing import Optional, List, Dict, Union, Callable
from collections import deque

import pandas
from strct.dicts import reverse_dict_partial

from pdpipe.core import PdPipelineStage, ColumnsBasedPipelineStage
# from pdpipe.util import out_of_place_col_insert
from pdpipe.shared import (
    _interpret_columns_param,
    _list_str,
)

import pdpipe.cond as cond
from pdpipe.types import ColumnsParamType
from pdpipe.exceptions import FailedConditionError
from pdpipe.cq import ColumnQualifier


class ColDrop(ColumnsBasedPipelineStage):
    """A pipeline stage that drops columns by name.

    Parameters
    ----------
    columns : single label, list-like or callable
        The label, or an iterable of labels, of columns to drop. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame (see `pdpipe.cq`).
    errors : {‘ignore’, ‘raise’}, default ‘raise’
        If ‘ignore’, suppress error and existing labels are dropped.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> pdp.ColDrop('num').apply(df)
      char
    1    a
    2    b
    """

    def __init__(
        self,
        columns: ColumnsParamType,
        errors: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        self._errors = errors
        self._post_cond = cond.HasNoColumn(columns)
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop columns {}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'error'
        super().__init__(**super_kwargs)

    def _prec(self, df: pandas.DataFrame) -> bool:
        if self._errors != 'ignore':
            return super()._prec(df)
        return True

    def _post(self, df: pandas.DataFrame) -> bool:
        return self._post_cond(df)

    def _transformation(
        self, df: pandas.DataFrame, verbose: bool, fit: bool,
    ) -> pandas.DataFrame:
        return df.drop(
            self._get_columns(df, fit=fit), axis=1, errors=self._errors)


class ValDrop(ColumnsBasedPipelineStage):
    """A pipeline stage that drops rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to drop.
    columns : single label, list-like or callable, default None
        The label, or an iterable of labels, of columns to check for the given
        values. Alternatively, this parameter can be assigned a callable
        returning an iterable of labels from an input pandas.DataFrame. See
        `pdpipe.cq`. If set to None, all columns are checked.
    exclude_columns : label, iterable or callable, optional
        The label, or an iterable of labels, of columns to exclude, given the
        `columns` parameter. Alternatively, this parameter can be assigned a
        callable returning a labels iterable from an input pandas.DataFrame.
        See `pdpipe.cq`. Optional. By default no columns are excluded.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,5],[18,11]], [1,2,3], ['a','b'])
    >>> pdp.ValDrop([4], 'a').apply(df)
        a   b
    1   1   4
    3  18  11
    >>> pdp.ValDrop([4]).apply(df)
        a   b
    3  18  11
    """

    def __init__(
        self,
        values: List[object],
        columns: ColumnsParamType = None,
        **kwargs: object,
    ) -> None:
        self._values = values
        self._values_str = _list_str(self._values)
        super_kwargs = {
            'columns': columns,
            'desc_temp': f'Drop values {self._values_str} in columns {{}}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'all'
        super().__init__(**super_kwargs)

    def _transformation(
        self, df: pandas.DataFrame, verbose: bool, fit: bool,
    ) -> pandas.DataFrame:
        inter_df = df
        before_count = len(inter_df)
        columns_to_check = self._get_columns(df, fit=fit)
        for col in columns_to_check:
            inter_df = inter_df[~inter_df[col].isin(self._values)]
        if verbose:
            print(f"{before_count - len(inter_df)} rows dropped.")
        return inter_df


class ValKeep(ColumnsBasedPipelineStage):
    """A pipeline stage that keeps rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to keep.
    columns : single label, list-like or callable, default None
        The label, or an iterable of labels, of columns to check for the given
        values. Alternatively, this parameter can be assigned a callable
        returning an iterable of labels from an input pandas.DataFrame. See
        `pdpipe.cq`. If set to None, all columns are checked.
    exclude_columns : single label, iterable or callable, optional
        The label, or an iterable of labels, of columns to exclude, given the
        `columns` parameter. Alternatively, this parameter can be assigned a
        callable returning a labels iterable from an input pandas.DataFrame.
        See `pdpipe.cq`. Optional. By default no columns are excluded.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
    >>> pdp.ValKeep([4, 5], 'a').apply(df)
       a   b
    2  4   5
    3  5  11
    >>> pdp.ValKeep([4, 5]).apply(df)
       a  b
    2  4  5
    """

    def __init__(self, values, columns=None, **kwargs):
        self._values = values
        self._values_str = _list_str(self._values)
        super_kwargs = {
            'columns': columns,
            'desc_temp': f'Keep values {self._values_str} in columns {{}}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'all'
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        inter_df = df
        before_count = len(inter_df)
        columns_to_check = self._get_columns(df, fit=fit)
        for col in columns_to_check:
            inter_df = inter_df[inter_df[col].isin(self._values)]
        if verbose:
            print(f"{before_count - len(inter_df)} rows dropped.")
        return inter_df


class ColRename(PdPipelineStage):
    """A pipeline stage that renames a column or columns.

    Parameters
    ----------
    rename_mapper : dict-like or callable
        Maps old column names to new ones.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> pdp.ColRename({'num': 'len', 'char': 'initial'}).apply(df)
       len initial
    1    8       a
    2    5       b

    >>> def renamer(lbl: str):
    ...    if lbl.startswith('n'):
    ...       return 'foo'
    ...    return lbl
    >>> pdp.ColRename(renamer).apply(df)
       foo char
    1    8    a
    2    5    b
    """

    _DEF_COLDRENAME_EXC_MSG = ("ColRename stage failed because not all columns"
                               " {} were found in input dataframe.")

    def __init__(self, rename_mapper: Union[Dict, Callable], **kwargs):
        self._rename_mapper = rename_mapper
        try:
            columns_str = _list_str(list(rename_mapper.keys()))
            mapper_repr = str(rename_mapper)
            keys_set = set(self._rename_mapper.keys())
            required_labels = list(keys_set)
            _tprec = cond.HasAllColumns(required_labels)
        except AttributeError:  # rename mapper is a callable
            mapper_repr = rename_mapper.__name__
            doc = rename_mapper.__doc__
            if doc is None:
                columns_str = f"by func {rename_mapper.__name__}"
            else:
                columns_str = (
                    f"by func {rename_mapper.__name__} with "
                    f"doc: {rename_mapper.__doc__}"
                )
            _tprec = cond.AlwaysTrue()
        try:
            suffix = 's' if len(rename_mapper) > 1 else ''
        except TypeError:
            suffix = 's'
        self._tprec = _tprec
        super_kwargs = {
            'exmsg': ColRename._DEF_COLDRENAME_EXC_MSG.format(columns_str),
            'desc': f"Rename column{suffix} with {mapper_repr}",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._tprec(df)

    def _transform(self, df, verbose):
        return df.rename(columns=self._rename_mapper)


class DropNa(PdPipelineStage):
    """A pipeline stage that drops null values.

    Supports all parameter supported by pandas.dropna function.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,None],[1,11]], [1,2,3], ['a','b'])
    >>> pdp.DropNa().apply(df)
       a     b
    1  1   4.0
    3  1  11.0
    """

    _DEF_DROPNA_EXC_MSG = "DropNa stage failed."
    _DROPNA_KWARGS = ['axis', 'how', 'thresh', 'subset', 'inplace']

    def __init__(self, **kwargs):
        common = set(kwargs.keys()).intersection(DropNa._DROPNA_KWARGS)
        self.dropna_kwargs = {key: kwargs.pop(key) for key in common}
        super_kwargs = {
            'exmsg': DropNa._DEF_DROPNA_EXC_MSG,
            'desc': "Drops null values."
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        before_count = len(df)
        ncols_before = len(df.columns)
        inter_df = df.dropna(**self.dropna_kwargs)
        if verbose:
            print(
                f"{before_count - len(inter_df)} rows, "
                f"{ncols_before - len(inter_df.columns)} columns dropeed"
            )
        return inter_df


class SetIndex(PdPipelineStage):
    """A pipeline stage that set existing columns as index.

    Supports all parameter supported by pandas.set_index function except for
    `inplace`.

    Example
    -------
    >> import pandas as pd; import pdpipe as pdp;
    >> df = pd.DataFrame([[1,4],[3, 11]], [1,2], ['a','b'])
    >> pdp.SetIndex('a').apply(df)
        b
    a
    1   4
    3  11
    """

    _DEF_SETIDX_EXC_MSG = "SetIndex stage failed."
    _DEF_SETIDX_APP_MSG = "Setting indexes..."
    _SETINDEX_KWARGS = ['drop', 'append', 'verify_integrity']

    def __init__(self, keys, **kwargs):
        common = set(kwargs.keys()).intersection(SetIndex._SETINDEX_KWARGS)
        self.setindex_kwargs = {key: kwargs.pop(key) for key in common}
        self.keys = keys
        if hasattr(keys, '__iter__') and not isinstance(keys, str):
            _tprec = cond.HasAllColumns(list(keys))
        else:
            _tprec = cond.HasAllColumns([keys])
        self._tprec = _tprec
        super_kwargs = {
            'exmsg': SetIndex._DEF_SETIDX_EXC_MSG,
            'desc': "Set indexes."
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._tprec(df)

    def _transform(self, df, verbose):
        return df.set_index(keys=self.keys, **self.setindex_kwargs)


class FreqDrop(PdPipelineStage):
    """A pipeline stage that drops rows by value frequency.

    Parameters
    ----------
    threshold : int
        The minimum frequency required for a value to be kept.
    column : str
        The name of the colum to check for the given value frequency.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,5],[1,11]], [1,2,3], ['a','b'])
    >>> pdp.FreqDrop(2, 'a').apply(df)
       a   b
    1  1   4
    3  1  11
    """

    _DEF_FREQDROP_EXC_MSG = ("FreqDrop stage failed because column {} was not"
                             " found in input dataframe.")
    _DEF_FREQDROP_DESC = "Drop values with frequency < {} in column {}."

    def __init__(self, threshold: int, column: str, **kwargs):
        self._threshold = threshold
        self._column = column
        super_kwargs = {
            'exmsg': FreqDrop._DEF_FREQDROP_EXC_MSG.format(self._column),
            'desc': FreqDrop._DEF_FREQDROP_DESC.format(
                self._threshold, self._column)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._column in df.columns

    def _transform(self, df, verbose):
        inter_df = df
        before_count = len(inter_df)
        valcount = df[self._column].value_counts()
        to_drop = valcount[valcount < self._threshold].index
        inter_df = inter_df[~inter_df[self._column].isin(to_drop)]
        if verbose:
            print(f"{before_count - len(inter_df)} rows dropped.")
        return inter_df


class ColReorder(PdPipelineStage):
    """A pipeline stage that reorders columns.

    Parameters
    ----------
    positions : dict
        A mapping of column names to their desired positions after reordering.
        Columns not included in the mapping will maintain their relative
        positions over the non-mapped colums.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,4,3,7]], columns=['a', 'b', 'c', 'd'])
    >>> pdp.ColReorder({'b': 0, 'c': 3}).apply(df)
       b  a  d  c
    0  4  8  7  3
    """

    _DEF_ORD_EXC_MSG = ("ColReorder stage failed because not all columns in {}"
                        " were found in input dataframe.")

    def __init__(self, positions, **kwargs):
        self._col_to_pos = positions
        self._pos_to_col = reverse_dict_partial(positions)
        super_kwargs = {
            'exmsg': ColReorder._DEF_ORD_EXC_MSG.format(self._col_to_pos),
            'desc': f"Reorder columns by {self._col_to_pos}",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._col_to_pos.keys()).issubset(df.columns)

    def _transform(self, df, verbose):
        cols = df.columns
        map_cols = list(self._col_to_pos.keys())
        non_map_cols = deque(x for x in cols if x not in map_cols)
        new_columns = []
        try:
            for pos in range(len(cols)):
                if pos in self._pos_to_col:
                    new_columns.append(self._pos_to_col[pos])
                else:
                    new_columns.append(non_map_cols.popleft())
            return df[new_columns]
        except (IndexError):
            raise ValueError(f"Bad positions mapping given: {new_columns}")


class RowDrop(ColumnsBasedPipelineStage):
    """A pipeline stage that drops rows by callable conditions.

    Parameters
    ----------
    conditions : list-like or dict
        The list of conditions that make a row eligible to be dropped. Each
        condition must be a callable that take a cell value and return a bool
        value. If a list of callables is given, the conditions are checked for
        each column value of each row. If a dict mapping column labels to
        callables is given, then each condition is only checked for the column
        values of the designated column.
    reduce : 'any', 'all' or 'xor', default 'any'
        Determines how row conditions are reduced. If set to 'all', a row must
        satisfy all given conditions to be dropped. If set to 'any', rows
        satisfying at least one of the conditions are dropped. If set to 'xor',
        rows satisfying exactly one of the conditions will be dropped. Set to
        'any' by default.
    columns : single label, iterable or callable, optional
        The label, or an iterable of labels, of columns. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`. If given,
        input conditions will be applied to the sub-dataframe made up of
        these columns to determine which rows to drop. Ignored if `conditions`
        is provided with a dict object. If `conditions` is a list and this
        parameter is not provided, all columns are checked (unless
        `exclude_columns` is additionally provided)
    exclude_columns : single label, iterable or callable, optional
        The label, or an iterable of labels, of columns to exclude, given the
        `columns` parameter. Alternatively, this parameter can be assigned a
        callable returning a labels iterable from an input pandas.DataFrame.
        See `pdpipe.cq`. Optional. By default no columns are excluded.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
    >>> pdp.RowDrop([lambda x: x < 2]).apply(df)
       a   b
    2  4   5
    3  5  11
    >>> pdp.RowDrop({'a': lambda x: x == 4}).apply(df)
       a   b
    1  1   4
    3  5  11
    """

    _REDUCERS = {
        'all': all,
        'any': any,
        'xor': lambda x: sum(x) == 1
    }

    class _DictRowCond(object):
        """Filter rows by a dict of conditions."""

        def __init__(self, conditions, reducer):
            self.conditions = conditions
            self.reducer = reducer

        def __call__(self, row):
            res = [cond(row[lbl]) for lbl, cond in self.conditions.items()]
            return self.reducer(res)

    class _ListRowCond(object):
        """Filter rows by a list of conditions."""

        def __init__(self, conditions, reducer):
            self.conditions = conditions
            self.reducer = reducer

        def __call__(self, row):
            res = [self.reducer(row.apply(cond)) for cond in self.conditions]
            return self.reducer(res)

    def _row_condition_builder(self, conditions, reduce):
        reducer = RowDrop._REDUCERS[reduce]
        if self._cond_is_dict:
            row_cond = RowDrop._DictRowCond(
                conditions=conditions, reducer=reducer)
        else:
            row_cond = RowDrop._ListRowCond(
                conditions=conditions, reducer=reducer)
        return row_cond

    def __init__(self, conditions, reduce=None, columns=None, **kwargs):
        self._conditions = conditions
        if reduce is None:
            reduce = 'any'
        self._reduce = reduce
        if reduce not in RowDrop._REDUCERS.keys():
            raise ValueError((
                "{} is an unsupported argument for the 'reduce' parameter of "
                "the RowDrop constructor!").format(reduce))
        self._cond_is_dict = isinstance(conditions, dict)
        if self._cond_is_dict:
            valid = all([callable(cond) for cond in conditions.values()])
            if not valid:
                raise ValueError(
                    "Condition dicts given to RowDrop must map to callables!")
            columns = list(conditions.keys())
        else:
            valid = all([callable(cond) for cond in conditions])
            if not valid:
                raise ValueError(
                    "RowDrop condition lists can contain only callables!")
        self._row_cond = self._row_condition_builder(conditions, reduce)
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop rows in columns {} by conditions',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'all'
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        before_count = len(df)
        columns = self._get_columns(df, fit=fit)
        subdf = df[columns]
        drop_index = ~subdf.apply(self._row_cond, axis=1)
        inter_df = df[drop_index]
        if verbose:
            print(f"{before_count - len(inter_df)} rows dropped.")
        return inter_df


class Schematize(PdPipelineStage):
    """Enforces a column schema on input dataframes.

    Parameters
    ----------
    columns: sequence of label, optional
        The dataframe schema to enforce on input dataframes. If set to None,
        the schema is learned in fit time and applied in subsequent transforms.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[2, 4, 8],[3, 6, 9]], [1, 2], ['a', 'b', 'c'])
    >>> pdp.Schematize(['a', 'c']).apply(df)
       a  c
    1  2  8
    2  3  9
    >>> pdp.Schematize(['c', 'b']).apply(df)
       c  b
    1  8  4
    2  9  6
    """

    def __init__(
        self,
        columns: Optional[List[object]],
        **kwargs: object,
    ) -> None:
        if columns is None:
            self._adaptive = True
            self._columns = None
            self._columns_str = '<Learnable Schema>'
            exmsg = "Learnable schematize failed in precondition unexpectedly!"
        else:
            self._adaptive = False
            self._columns = _interpret_columns_param(columns)
            self._columns_str = _list_str(self._columns)
            exmsg = (
                f"Not all required columns {self._columns_str} "
                f"found in input dataframe!"
            )
        desc = (
            f"Transform input dataframes to the following schema: "
            f"{self._columns_str}"
        )
        super_kwargs = {
            'exmsg': exmsg,
            'desc': desc,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df: pandas.DataFrame) -> bool:
        if self._adaptive and not self.is_fitted:
            return True
        return set(self._columns).issubset(df.columns)

    def _transform(
            self, df: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
        return df[self._columns]

    def _fit_transform(
            self, df: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
        if self._adaptive:
            self._columns = df.columns
            self.is_fitted = True
            return df
        return df[self._columns]


class DropDuplicates(ColumnsBasedPipelineStage):
    """Drop duplicates in the given columns.

    Parameters
    ----------
    columns: column label or sequence of labels, optional
        The labels of the columns to consider for duplication drop. If not
        populated, duplicates are dropped from all columns.
    exclude_columns : object, iterable or callable, optional
        The label, or an iterable of labels, of columns to exclude, given the
        `columns` parameter. Alternatively, this parameter can be assigned a
        callable returning a labels iterable from an input pandas.DataFrame.
        See `pdpipe.cq`. Optional. By default no columns are excluded.

    Examples
    --------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[8, 1],[8, 2], [9, 2]], [1,2,3], ['a', 'b'])
        >>> pdp.DropDuplicates('a').apply(df)
           a  b
        1  8  1
        3  9  2
    """

    def __init__(self, columns=None, **kwargs):
        super_kwargs = {
            'columns': columns,
            'desc_temp': 'Drop duplicates in columns {}',
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'all'
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        columns = self._get_columns(df, fit=fit)
        inter_df = df.drop_duplicates(subset=columns)
        if verbose:
            print(f"{len(df) - len(inter_df)} rows dropped.")
        return inter_df


class ColumnDtypeEnforcer(PdPipelineStage):
    """A pipeline stage enforcing column dtypes.

    Parameters
    ----------
    column_to_dtype: dict of labels / ColumnQualifiers to dtypes
        Use {col: dtype, …}, where col is a column label and dtype is a
        numpy.dtype or Python type to cast one or more of the DataFrame’s
        columns to column-specific types. Alternatively, you can provide
        `ColumnQualifier` objects as keys. If at least one such key is present,
        the lbl-to-dtype dict is dynamically inferred each time the pipeline
        stage is applied (note that `ColumnQualifier` objects are fittable by
        default, so to have column labels re-inferred after the first stage
        application you'll have to set `fittable=False` for the
        `ColumnQualifier` you use, see `pdpipe.cq`).
    errors: {‘raise’, ‘ignore’}, default ‘raise’
        Control raising of exceptions on invalid data for provided dtype.
        - raise : allow exceptions to be raised
        - ignore : suppress exceptions. On error return original object.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'initial'])
    >>> pdp.ColumnDtypeEnforcer({'num': float}).apply(df)
       num initial
    1  8.0       a
    2  5.0       b

    >>> pdp.ColumnDtypeEnforcer({pdp.cq.StartWith('n'): float}).apply(df)
       num initial
    1  8.0       a
    2  5.0       b
    """

    _DEF_COL_DTYPE_ENF_EXC_MSG = (
        "ColumnDtypeEnforcer stage failed because not all columns"
        " {} were found in input dataframe.")

    def __init__(
        self,
        column_to_dtype: Dict,
        errors: Optional[str] = 'raise',
        **kwargs: object,
    ) -> None:
        # if none of the keys in column_to_dtype is a ColumnQualifier
        if not any(isinstance(
                x, ColumnQualifier) for x in column_to_dtype.keys()):
            # its a static map; use it as is
            self._column_to_dtype = column_to_dtype
            keys_set = set(column_to_dtype.keys())
            _tprec = cond.HasAllColumns(list(keys_set))
        else:
            # else, it's at least partly dynamic, and will have to infer it
            # on run time
            self._dynamic_column_to_dtype = column_to_dtype
            _tprec = cond.AlwaysTrue()
        self._tprec = _tprec
        self._errors = errors
        columns_str = _list_str(list(column_to_dtype.keys()))
        suffix = 's' if len(column_to_dtype) > 1 else ''
        super_kwargs = {
            'exmsg': ColumnDtypeEnforcer._DEF_COL_DTYPE_ENF_EXC_MSG.format(
                columns_str),
            'desc': f"Enforce column{suffix} dtype with {column_to_dtype}",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _col_to_dtype_from_df(self, df: pandas.DataFrame) -> Dict:
        try:
            return self._column_to_dtype
        except AttributeError:
            column_to_dtype = {}
            for k, dtype in self._dynamic_column_to_dtype.items():
                try:
                    column_to_dtype.update({
                        lbl: dtype
                        for lbl in k(df)
                    })
                except TypeError:  # k is not a callable
                    column_to_dtype[k] = dtype
            return column_to_dtype

    def _prec(self, df: pandas.DataFrame) -> bool:
        return self._tprec(df)

    def _transform(
        self,
        df: pandas.DataFrame,
        verbose: bool,
    ) -> pandas.DataFrame:
        lbl_to_dtype = self._col_to_dtype_from_df(df)
        return df.astype(
            dtype=lbl_to_dtype,
            copy=True,
            errors=self._errors,
        )


class ConditionValidator(PdPipelineStage):
    """A pipeline stage that validates boolean conditions on dataframes.

    The stage does not change the input dataframe in any way.

    The constructor expects either a single callable or a list-like of callable
    objects, and checks that all these callable return True - meaning all
    defined conditions hold - for input dataframes.

    Naturally, pdpipe `Condition` objects from the `pdpipe.cond` module can be used.

    Parameters
    ----------
    conditions : callable or list-like of callable
        The conditions to check for input dataframes. Naturally, pdpipe
        `Condition` objects from the `pdpipe.cond` module can be used.
    reducer : callable, optional
        The callable that reduces the list of boolean result to a single
        result. By default the built-in `all` function is used, so all
        conditions must hold for this pipeline stage to validate an input
        dataframe. The built-in `any` function may be used to validate at least
        one condition holds, and of course custom reducing functions can be
        used.
    errors : str, default 'raise'
        If set to 'raise', the default, then if the result boolean result is
        False a FailedConditionError is raised on stage application. If set to
        'ignore', then conditions are checked, the results are printed if the
        application was called with `verbose=True`, and pipeline application
        continues. Any other value is interpreted as 'raise'.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,None],[1,11]], [1,2,3], ['a','b'])
    >>> pdp.ConditionValidator(lambda df: len(df.columns) == 5).apply(df)
    Traceback (most recent call last):
       ...
    pdpipe.exceptions.FailedConditionError: ConditionValidator stage failed; some conditions did not hold for the input dataframe!

    >>> pdp.ConditionValidator(pdp.cond.HasNoMissingValues()).apply(df)
    Traceback (most recent call last):
       ...
    pdpipe.exceptions.FailedConditionError: ConditionValidator stage failed; some conditions did not hold for the input dataframe!
    """  # noqa: E501

    def __init__(
        self,
        conditions: Union[Callable, List[Callable]],
        reducer: Optional[Callable] = all,
        errors: Optional[str] = 'raise',
        **kwargs: object,
    ):
        if callable(conditions):
            conditions = [conditions]
        self._conditions = conditions
        self._reducer = reducer
        self._errors = 'raise'
        self._raise = True
        if errors == 'ignore':
            self._errors = 'ignore'
            self._raise = False
        super_kwargs = {
            'desc': "Validates conditions"
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        results = []
        for cond_obj in self._conditions:
            try:
                res = cond_obj(df)
            except Exception as e:
                raise ValueError(
                    f"Supplied condition raised a {e} exception when applied "
                    "to input dataframe!"
                ) from e
            if verbose and not res:
                cond_repr = cond_obj.__doc__
                if cond_repr is None:
                    cond_repr = str(cond_obj)
                print(f"  + Condition failed for input dataframe: {cond_repr}")
            results.append(res)
        reduced_result = self._reducer(results)
        if self._raise and not reduced_result:
            raise FailedConditionError(
                "ConditionValidator stage failed; some conditions did not hold"
                " for the input dataframe!")
        return df
