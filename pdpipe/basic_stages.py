"""Basic pdpipe PiplineStages."""

import types

import pandas as pd
import sortedcontainers as sc
import tqdm

from pdpipe.core import PipelineStage
from pdpipe.util import out_of_place_col_insert

from pdpipe.shared import (
    _interpret_columns_param,
    _list_str
)


class ColDrop(PipelineStage):
    """A pipline stage that drops columns by name.

    Parameters
    ----------
    columns : str, iterable or callable
        The name, or an iterable of names, of columns to drop. Alternatively,
        columns can be assigned a callable returning bool values for
        pandas.Series objects; if this is the case, every column for which it
        return True will be dropped.

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

    def __init__(self, columns, **kwargs):
        self._columns = columns
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
        return set(self._columns).issubset(df.columns)

    def _op(self, df, verbose):
        if callable(self._columns):
            cols_to_drop = [
                col for col in df.columns
                if self._columns(df[col])
            ]
            return df.drop(cols_to_drop, axis=1)
        return df.drop(self._columns, axis=1)


class ValDrop(PipelineStage):
    """A pipline stage that drops rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to drop.
    columns : str or list-like, defualt None
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
        columns_to_check = self._columns
        if self._columns is None:
            columns_to_check = df.columns
        for col in columns_to_check:
            inter_df = inter_df[~inter_df[col].isin(self._values)]
        return inter_df


class ValKeep(PipelineStage):
    """A pipline stage that keeps rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to keep.
    columns : str or list-like, defualt None
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
    """A pipline stage that renames a column or columns.

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
