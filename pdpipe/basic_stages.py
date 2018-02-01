"""Basic pdpipe PipelineStages."""

import types


from pdpipe.core import PipelineStage
# from pdpipe.util import out_of_place_col_insert

from pdpipe.shared import (
    _interpret_columns_param,
    _list_str
)


class ColDrop(PipelineStage):
    """A pipeline stage that drops columns by name.

    Parameters
    ----------
    columns : str, iterable or callable
        The name, or an iterable of names, of columns to drop. Alternatively,
        columns can be assigned a callable returning bool values for
        pandas.Series objects; if this is the case, every column for which it
        return True will be dropped.
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

    _DEF_COLDROP_EXC_MSG = ("ColDrop stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_COLDROP_APPLY_MSG = 'Dropping columns {}...'

    def _default_desc(self):
        if isinstance(self._columns, types.FunctionType):
            return "Drop columns by lambda."
        return "Drop column{} {}".format(
            's' if len(self._columns) > 1 else '', self._columns_str)

    def __init__(self, columns, errors=None, **kwargs):
        self._columns = columns
        self._errors = errors
        self._columns_str = _list_str(self._columns)
        if not callable(columns):
            self._columns = _interpret_columns_param(columns, 'columns')
        super_kwargs = {
            'exmsg': ColDrop._DEF_COLDROP_EXC_MSG.format(self._columns_str),
            'appmsg': ColDrop._DEF_COLDROP_APPLY_MSG.format(self._columns_str),
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        if callable(self._columns):
            return True
        if self._errors != 'ignore':
            return set(self._columns).issubset(df.columns)
        return True

    def _op(self, df, verbose):
        if callable(self._columns):
            cols_to_drop = [
                col for col in df.columns
                if self._columns(df[col])
            ]
            return df.drop(cols_to_drop, axis=1, errors=self._errors)
        return df.drop(self._columns, axis=1, errors=self._errors)


class ValDrop(PipelineStage):
    """A pipeline stage that drops rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to drop.
    columns : str or list-like, default None
        The name, or an iterable of names, of columns to check for the given
        values. If set to None, all columns are checked.

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

    _DEF_VALDROP_EXC_MSG = ("ValDrop stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_VALDROP_APPLY_MSG = "Dropping values {}..."

    def _default_desc(self):
        if self._columns:
            return "Drop values {} in column{} {}".format(
                self._values_str, 's' if len(self._columns) > 1 else '',
                self._columns_str)
        return "Drop values {}".format(self._values_str)

    def __init__(self, values, columns=None, **kwargs):
        self._values = values
        self._values_str = _list_str(self._values)
        self._columns_str = _list_str(columns)
        if columns is None:
            self._columns = None
            apply_msg = ValDrop._DEF_VALDROP_APPLY_MSG.format(
                self._values_str)
        else:
            self._columns = _interpret_columns_param(columns, 'columns')
            apply_msg = ValDrop._DEF_VALDROP_APPLY_MSG.format(
                "{} in {}".format(
                    self._values_str, self._columns_str))
        super_kwargs = {
            'exmsg': ValDrop._DEF_VALDROP_EXC_MSG.format(self._columns_str),
            'appmsg': apply_msg,
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _op(self, df, verbose):
        inter_df = df
        before_count = len(inter_df)
        columns_to_check = self._columns
        if self._columns is None:
            columns_to_check = df.columns
        for col in columns_to_check:
            inter_df = inter_df[~inter_df[col].isin(self._values)]
        if verbose:
            print("{} rows dropped.".format(before_count - len(inter_df)))
        return inter_df


class ValKeep(PipelineStage):
    """A pipeline stage that keeps rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to keep.
    columns : str or list-like, default None
        The name, or an iterable of names, of columns to check for the given
        values. If set to None, all columns are checked.

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

    _DEF_VALKEEP_EXC_MSG = ("ValKeep stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_VALKEEP_APPLY_MSG = "Keeping values {}..."

    def _default_desc(self):
        if self._columns:
            return "Keep values {} in column{} {}".format(
                self._values_str, 's' if len(self._columns) > 1 else '',
                self._columns_str)
        return "Keep values {}".format(self._values_str)

    def __init__(self, values, columns=None, **kwargs):
        self._values = values
        self._values_str = _list_str(self._values)
        self._columns_str = _list_str(columns)
        if columns is None:
            self._columns = None
            apply_msg = ValKeep._DEF_VALKEEP_APPLY_MSG.format(
                self._values_str)
        else:
            self._columns = _interpret_columns_param(columns, 'columns')
            apply_msg = ValKeep._DEF_VALKEEP_APPLY_MSG.format(
                "{} in {}".format(
                    self._values_str, self._columns_str))
        super_kwargs = {
            'exmsg': ValKeep._DEF_VALKEEP_EXC_MSG.format(self._columns_str),
            'appmsg': apply_msg,
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _op(self, df, verbose):
        inter_df = df
        columns_to_check = self._columns
        if self._columns is None:
            columns_to_check = df.columns
        for col in columns_to_check:
            inter_df = inter_df[inter_df[col].isin(self._values)]
        return inter_df


class ColRename(PipelineStage):
    """A pipeline stage that renames a column or columns.

    Parameters
    ----------
    rename_map : dict
        Maps old column names to new ones.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> pdp.ColRename({'num': 'len', 'char': 'initial'}).apply(df)
       len initial
    1    8       a
    2    5       b
    """

    _DEF_COLDRENAME_EXC_MSG = ("ColRename stage failed because not all columns"
                               " {} were found in input dataframe.")
    _DEF_COLDRENAME_APP_MSG = "Renaming column{} {}..."

    def __init__(self, rename_map, **kwargs):
        self._rename_map = rename_map
        columns_str = _list_str(list(rename_map.keys()))
        suffix = 's' if len(rename_map) > 1 else ''
        super_kwargs = {
            'exmsg': ColRename._DEF_COLDRENAME_EXC_MSG.format(columns_str),
            'appmsg': ColRename._DEF_COLDRENAME_APP_MSG.format(
                suffix, columns_str),
            'desc': "Rename column{} with {}".format(suffix, self._rename_map)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._rename_map.keys()).issubset(df.columns)

    def _op(self, df, verbose):
        return df.rename(columns=self._rename_map)


class DropNa(PipelineStage):
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
    _DEF_DROPNA_APP_MSG = "Dropping null values..."
    _DROPNA_KWARGS = ['axis', 'how', 'thresh', 'subset', 'inplace']

    def __init__(self, **kwargs):
        common = set(kwargs.keys()).intersection(DropNa._DROPNA_KWARGS)
        self.dropna_kwargs = {key: kwargs.pop(key) for key in common}
        super_kwargs = {
            'exmsg': DropNa._DEF_DROPNA_EXC_MSG,
            'appmsg': DropNa._DEF_DROPNA_APP_MSG,
            'desc': "Drops null values."
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _op(self, df, verbose):
        before_count = len(df)
        inter_df = df.dropna(**self.dropna_kwargs)
        if verbose:
            print("{} rows dropeed".format(before_count - len(inter_df)))
        return inter_df


class FreqDrop(PipelineStage):
    """A pipeline stage that drops rows by value frequency.

    Parameters
    ----------
    threshold : int
        The minimum frequency required for a value to be kept.
    column : str
        The name of the colums to check for the given value frequency.

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
    _DEF_FREQDROP_APPLY_MSG = ("Dropping values with frequency < {} in column"
                               " {}...")
    _DEF_FREQDROP_DESC = "Drop values with frequency < {} in column {}."

    def __init__(self, threshold, column, **kwargs):
        self._threshold = threshold
        self._column = column
        apply_msg = FreqDrop._DEF_FREQDROP_APPLY_MSG.format(
            self._threshold, self._column)
        super_kwargs = {
            'exmsg': FreqDrop._DEF_FREQDROP_EXC_MSG.format(self._column),
            'appmsg': apply_msg,
            'desc': FreqDrop._DEF_FREQDROP_DESC.format(
                self._threshold, self._column)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._column in df.columns

    def _op(self, df, verbose):
        inter_df = df
        before_count = len(inter_df)
        valcount = df[self._column].value_counts()
        to_drop = valcount[valcount < self._threshold].index
        inter_df = inter_df[~inter_df[self._column].isin(to_drop)]
        if verbose:
            print("{} rows dropped.".format(before_count - len(inter_df)))
        return inter_df
