"""Basic pdpipe PiplineStages."""

import types

import pandas as pd
import sortedcontainers as sc
import sklearn.preprocessing
import tqdm

from pdpipe.core import PipelineStage
from pdpipe.util import out_of_place_col_insert


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


def _list_str(listi):
    if listi is None:
        return None
    if isinstance(listi, (list, tuple)):
        return ', '.join([str(elem) for elem in listi])
    return listi


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


class Bin(PipelineStage):
    """A pipline stage that adds a binned version of a column or columns.

    If drop is set to True the new columns retain the names of the source
    columns; otherwise, the resulting column gain the suffix '_bin'

    Parameters
    ----------
    bin_map : dict
        Maps column names to bin arrays. The bin array is interpreted as
        containing start points of consecutive bins, except for the final
        point, assumed to be the end point of the last bin. Additionally, a
        bin array implicitly projects a left-most bin containing all elements
        smaller than the left-most end point and a right-most bin containing
        all elements larger that the right-most end point. For example, the
        list [0, 5, 8] is interpreted as the bins (-∞, 0),
        [0-5), [5-8) and [5, ∞).
    drop : bool, default True
        If set to True, the source columns are dropped after being binned.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[-3],[4],[5], [9]], [1,2,3, 4], ['speed'])
    >>> pdp.Bin({'speed': [5]}, drop=False).apply(df)
       speed speed_bin
    1     -3        <5
    2      4        <5
    3      5        5≤
    4      9        5≤
    >>> pdp.Bin({'speed': [0,5,8]}, drop=False).apply(df)
       speed speed_bin
    1     -3        <0
    2      4       0-5
    3      5       5-8
    4      9        8≤
    """

    _DEF_BIN_EXC_MSG = ("Bin stage failed because not all columns "
                        "{} were found in input dataframe.")
    _DEF_BIN_APP_MSG = "Binning column{} {}..."

    def _default_desc(self):
        string = ""
        columns = list(self._bin_map.keys())
        col1 = columns[0]
        string += "Bin {} by {},\n".format(col1, self._bin_map[col1])
        for col in columns:
            string += "bin {} by {},\n".format(col, self._bin_map[col])
        string = string[0:-2] + '.'
        return string

    def __init__(self, bin_map, drop=True, **kwargs):
        self._bin_map = bin_map
        self._drop = drop
        columns_str = _list_str(list(bin_map.keys()))
        super_kwargs = {
            'exmsg': Bin._DEF_BIN_EXC_MSG.format(columns_str),
            'appmsg': Bin._DEF_BIN_APP_MSG.format(
                's' if len(bin_map) > 1 else '', columns_str),
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._bin_map.keys()).issubset(df.columns)

    @staticmethod
    def _get_col_binner(bin_list):
        sorted_bins = sc.SortedList(bin_list)
        last_ix = len(sorted_bins) - 1
        def _col_binner(val):
            if val in sorted_bins:
                ind = sorted_bins.bisect(val)-1
                if ind == last_ix:
                    return '{}≤'.format(sorted_bins[-1])
                return '{}-{}'.format(sorted_bins[ind], sorted_bins[ind+1])
            try:
                ind = sorted_bins.bisect(val)
                if ind == 0:
                    return '<{}'.format(sorted_bins[ind])
                return '{}-{}'.format(sorted_bins[ind-1], sorted_bins[ind])
            except IndexError:
                return '{}≤'.format(sorted_bins[sorted_bins.bisect(val)-1])
        return _col_binner

    def _op(self, df, verbose):
        inter_df = df
        colnames = list(self._bin_map.keys())
        if verbose:
            colnames = tqdm.tqdm(colnames)
        for colname in colnames:
            if verbose:
                colnames.set_description(colname)
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_bin"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.apply(
                    self._get_col_binner(self._bin_map[colname])),
                loc=loc,
                column_name=new_name)
        return inter_df


class Binarize(PipelineStage):
    """A pipline stage that binarizes categorical columns.

    By default only k-1 dummies are created fo k categorical levels, as to
    avoid perfect multicollinearity between the dummy features (also called
    the dummy variable  trap). This is done since features are usually
    binarized for use with linear model, which require this behaviour.

    Parameters
    ----------
    columns : str or list-like, default None
        Column names in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be converted, except
        those given in the exclude_columns parameter.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    exclude_columns : str or list-like, default None
        Name or names of categorical columns to be excluded from binarization
        when the columns parameter is not given. If None no column is excluded.
        Ignored if the columns parameter is given.
    drop_first : bool, default True
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    drop : bool, default True
        If set to True, the source columns are dropped after being binarized.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([['USA'], ['UK'], ['Greece']], [1,2,3], ['Born'])
    >>> pdp.Binarize().apply(df)
       Born_UK  Born_USA
    1        0         1
    2        1         0
    3        0         0
    """

    _DEF_BINAR_EXC_MSG = ("Binarize stage failed because not all columns "
                          "{} were found in input dataframe.")
    _DEF_BINAR_APP_MSG = "Binarizing {}..."

    def __init__(self, columns=None, dummy_na=False, exclude_columns=None,
                 drop_first=True, drop=True, **kwargs):
        if columns is None:
            self._columns = None
        else:
            self._columns = _interpret_columns_param(columns, 'columns')
        self._dummy_na = dummy_na
        if exclude_columns is None:
            self._exclude_columns = []
        else:
            self._exclude_columns = _interpret_columns_param(
                exclude_columns, 'exclude_columns')
        self._drop_first = drop_first
        self._drop = drop
        col_str = _list_str(self._columns)
        super_kwargs = {
            'exmsg': Binarize._DEF_BINAR_EXC_MSG.format(col_str),
            'appmsg': Binarize._DEF_BINAR_APP_MSG.format(
                col_str or "all columns"),
            'desc': "Binarize {}".format(col_str or "all categorical columns")
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _op(self, df, verbose):
        columns_to_binar = self._columns
        if self._columns is None:
            columns_to_binar = list(set(df.select_dtypes(
                include=['object', 'category']).columns).difference(
                    self._exclude_columns))
        assign_map = {}
        if verbose:
            columns_to_binar = tqdm.tqdm(columns_to_binar)
        for colname in columns_to_binar:
            if verbose:
                columns_to_binar.set_description(colname)
            dummies = pd.get_dummies(
                df[colname], drop_first=False, dummy_na=self._dummy_na,
                prefix=colname, prefix_sep='_')
            nan_col = colname+".nan"
            if self._drop_first:
                if nan_col in dummies:
                    dummies.drop(nan_col, axis=1, inplace=True)
                else:
                    dummies.drop(dummies.columns[0], axis=1, inplace=True)
            for column in dummies:
                assign_map[column] = dummies[column]

        inter_df = df.assign(**assign_map)
        if self._drop:
            return inter_df.drop(columns_to_binar, axis=1)
        return inter_df


class MapColVals(PipelineStage):
    """A pipline stage that replaces the values of a column by a map.

    Parameters
    ----------
    columns : str or list-like
        Column names in the DataFrame to be encoded.
    value_map : dict, function or pandas.Series
        A dictionary mapping existing values to new ones. Not all existing
        values need to be referenced; missing one will neither be changed nor
        dropped. If a function is given, it is applied element-wise to given
        columns. If a Series is given, values are mapped by its index to its
        values.
    result_columns : str or list-like, default None
        The name of the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the name of the source column is used;
        otherwise, the name of the source column is used with the suffix
        '_map'.
    drop : bool, default True
        If set to True, the source column is dropped after being mapped.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1], [3], [2]], ['UK', 'USSR', 'US'], ['Medal'])
    >>> value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    >>> pdp.MapColVals('Medal', value_map).apply(df)
           Medal
    UK      Gold
    USSR  Bronze
    US    Silver
    """

    _DEF_MAP_COLVAL_EXC_MSG = ("MapColVals stage failed because column{} "
                               "{} were not found in input dataframe.")
    _DEF_MAP_COLVAL_APP_MSG = "Mapping values of column{} {} with {}..."

    def __init__(self, columns, value_map, result_columns=None,
                 drop=True, **kwargs):
        self._columns = _interpret_columns_param(columns, 'columns')
        self._value_map = value_map
        if result_columns is None:
            if drop:
                self._result_columns = self._columns
            else:
                self._result_columns = [col + '_map' for col in self._columns]
        else:
            self._result_columns = _interpret_columns_param(
                result_columns, 'result_columns')
            if len(self._result_columns) != len(self._columns):
                raise ValueError("columns and result_columns parameters must"
                                 " be string lists of the same length!")
        col_str = _list_str(self._columns)
        sfx = 's' if len(self._columns) > 1 else ''
        self._drop = drop
        super_kwargs = {
            'exmsg': MapColVals._DEF_MAP_COLVAL_EXC_MSG.format(sfx, col_str),
            'appmsg': MapColVals._DEF_MAP_COLVAL_APP_MSG.format(
                sfx, col_str, self._value_map),
            'desc': "Map values of column{} {} with {}.".format(
                sfx, col_str, self._value_map)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns).issubset(df.columns)

    def _op(self, df, verbose):
        inter_df = df
        for i, colname in enumerate(self._columns):
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = self._result_columns[i]
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.map(self._value_map),
                loc=loc,
                column_name=new_name)
        return inter_df


class Encode(PipelineStage):
    """A pipline stage that encodes categorical columns to integer values.

    The encoder for each column is saved in the attribute 'encoders', which
    is a dict mapping each encoded column name to the
    sklearn.preprocessing.LabelEncoder object used to encode it.

    Parameters
    ----------
    columns : str or list-like, default None
        Column names in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be encoded, except
        those given in the exclude_columns parameter.
    exclude_columns : str or list-like, default None
        Name or names of categorical columns to be excluded from encoding
        when the columns parameter is not given. If None no column is excluded.
        Ignored if the columns parameter is given.
    drop : bool, default True
        If set to True, the source columns are dropped after being encoded,
        and the resulting encoded columns retain the names of the source
        colunmns. Otherwise, encoded columns gain the suffix '_enc'.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
    >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
    >>> encode_stage = pdp.Encode("lbl")
    >>> encode_stage(df)
         ph  lbl
    1   3.2    0
    2   7.2    1
    3  12.1    1
    >>> encode_stage.encoders["lbl"].inverse_transform([0,1,1])
    array(['acd', 'alk', 'alk'], dtype=object)
    """

    _DEF_ENCODE_EXC_MSG = ("Encode stage failed because not all columns "
                           "{} were found in input dataframe.")
    _DEF_ENCODE_APP_MSG = "Encoding {}..."

    def __init__(self, columns=None, exclude_columns=None, drop=True,
                 **kwargs):
        if columns is None:
            self._columns = None
        else:
            self._columns = _interpret_columns_param(columns, 'columns')
        if exclude_columns is None:
            self._exclude_columns = []
        else:
            self._exclude_columns = _interpret_columns_param(
                exclude_columns, 'exclude_columns')
        self._drop = drop
        self.encoders = {}
        col_str = _list_str(self._columns)
        super_kwargs = {
            'exmsg': Encode._DEF_ENCODE_EXC_MSG.format(col_str),
            'appmsg': Encode._DEF_ENCODE_APP_MSG.format(col_str),
            'desc': "Encode {}".format(col_str or "all categorical columns")
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _op(self, df, verbose):
        columns_to_encode = self._columns
        if self._columns is None:
            columns_to_encode = list(set(df.select_dtypes(
                include=['object', 'category']).columns).difference(
                    self._exclude_columns))
        if verbose:
            columns_to_encode = tqdm.tqdm(columns_to_encode)
        inter_df = df
        for colname in columns_to_encode:
            lbl_enc = sklearn.preprocessing.LabelEncoder()
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_enc"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=lbl_enc.fit_transform(source_col),
                loc=loc,
                column_name=new_name)
            self.encoders[colname] = lbl_enc
        return inter_df


class ColByFunc(PipelineStage):
    """A pipline stage generating a column by applying a function to each row.

    Parameters
    ----------
    func : function
        The function to be applied to each row of the processed DataFrame.
    result_columns : str or list-like
        The name of the new columns resulting from the function application.
        A name must be provided for each new column created, in their
        respective order.
    follow_column : str, default None,
        Resulting columns will be inserted after this column. If None, new
        columns are inserted at the end of the processed DataFrame.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.
    prec : function, default None
        A function taking a DataFrame, returning True if it this stage is
        applicable to the given DataFrame. If None is given, a function always
        returning True is used.


    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
    >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
    >>> encode_stage = pdp.Encode("lbl")
    >>> encode_stage(df)
         ph  lbl
    1   3.2    0
    2   7.2    1
    3  12.1    1
    >>> encode_stage.encoders["lbl"].inverse_transform([0,1,1])
    array(['acd', 'alk', 'alk'], dtype=object)
    """

    _DEF_COLBYFUNC_EXC_MSG = "Generating a column with a function {} failed."
    _DEF_COLBYFUNC_APP_MSG = "Generating a column with a function {}..."

    def __init__(self, func, result_columns, follow_column=None,
                 func_desc=None, prec=None, **kwargs):
        if func_desc is None:
            func_desc = ""
        if prec is None:
            prec = lambda df: True
        self._func = func
        self._result_columns = _interpret_columns_param(
            result_columns, 'result_columns')
        self._follow_column = follow_column
        self._func_desc = func_desc
        self._prec_func = prec
        super_kwargs = {
            'exmsg': Encode._DEF_COLBYFUNC_EXC_MSG.format(func_desc),
            'appmsg': Encode._DEF_COLBYFUNC_APP_MSG.format(func_desc),
            'desc': "Generating a column with a function {}.".format(
                self._func_desc)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._prec_func(df)

    def _op(self, df, verbose):
        new_cols = df.apply(self._func, axis=1)
        if isinstance(new_cols, pd.Series):
            loc = len(df.columns)
            if self._follow_column:
                loc = df.columns.get_loc(self._follow_column) + 1
            return out_of_place_col_insert(
                df=df,
                series=new_cols,
                loc=loc,
                column_name=self._result_columns[0])
        assign_map = {
            colname : new_cols[new_cols.columns[i]]
            for i, colname in enumerate(self._result_columns)
        }
        return df.assign(**assign_map)
