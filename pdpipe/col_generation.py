"""Column generation pdpipe PdPipelineStages."""

import abc
from typing import Union, Tuple, Optional, Dict, Callable

import numpy as np
import pandas as pd
import sortedcontainers as sc
import tqdm

from pdpipe.core import PdPipelineStage, ColumnsBasedPipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.cq import OfDtypes
from pdpipe.types import ColumnsParamType, ColumnLabelsType

from pdpipe.shared import _interpret_columns_param, _list_str

from .exceptions import PipelineApplicationError


class Bin(PdPipelineStage):
    """A pipeline stage that adds a binned version of a column or columns.

    If drop is set to True, the new columns retain the names of the source
    columns; otherwise, the resulting column gain the suffix '_bin'

    Parameters
    ----------
    bin_map : dict
        Maps column labels to bin arrays. The bin array is interpreted as
        containing start points of consecutive bins, except for the final
        point, assumed to be the end point of the last bin. Additionally, a
        bin array implicitly projects a left-most bin containing all elements
        smaller than the left-most end point and a right-most bin containing
        all elements larger that the right-most end point. For example, the
        list [0, 5, 8] is interpreted as the bins (-∞, 0), [0-5), [5-8) and
        [8, ∞).
    drop : bool, default True
        If set to True, the source columns are dropped after being binned.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[-3],[4],[5],[9]], [1,2,3,4], ['speed'])
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

    _DEF_BIN_EXC_MSG = (
        "Bin stage failed because not all columns "
        "{} were found in input dataframe."
    )

    def _default_desc(self):
        string = ""
        columns = list(self._bin_map.keys())
        col1 = columns[0]
        string += f"Bin {col1} by { self._bin_map[col1]},\n"
        for col in columns[1:]:
            string += f"bin {col} by {self._bin_map[col]},\n"
        string = string[0:-2] + "."
        return string

    def __init__(self, bin_map, drop=True, **kwargs):
        self._bin_map = bin_map
        self._drop = drop
        columns_str = _list_str(list(bin_map.keys()))
        super_kwargs = {
            "exmsg": Bin._DEF_BIN_EXC_MSG.format(columns_str),
            "desc": self._default_desc(),
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
                ind = sorted_bins.bisect(val) - 1
                if ind == last_ix:
                    return f"{sorted_bins[-1]}≤"
                return f"{sorted_bins[ind]}-{sorted_bins[ind + 1]}"
            try:
                ind = sorted_bins.bisect(val)
                if ind == 0:
                    return f"<{sorted_bins[ind]}"
                return f"{sorted_bins[ind - 1]}-{sorted_bins[ind]}"
            except IndexError:
                return f"{sorted_bins[sorted_bins.bisect(val) - 1]}≤"

        return _col_binner

    def _transform(self, df, verbose):
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
                    self._get_col_binner(self._bin_map[colname])
                ),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


class OneHotEncode(ColumnsBasedPipelineStage):
    """A pipeline stage that one-hot-encodes categorical columns.

    By default only k-1 dummies are created fo k categorical levels, as to
    avoid perfect multicollinearity between the dummy features (also called
    the dummy variable trap). This is done since features are usually one-hot
    encoded for use with linear models, which require this behaviour.

    Parameters
    ----------
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be converted, except
        those given in the exclude_columns parameter. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    exclude_columns : single label, list-like or callable, default None
        Label or labels of columns to be excluded from encoding. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
        pandas.DataFrame. See `pdpipe.cq`. Optional.
    drop_first : bool or single label, default True
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level. If a non bool argument matching one of the categories is
        provided, the dummy column corresponding to this value is dropped
        instead of the first level; if it matches no category the first
        category will still be dropped.
    drop : bool, default True
        If set to True, the source columns are dropped after being encoded.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([['USA'], ['UK'], ['Greece']], [1,2,3], ['Born'])
        >>> pdp.OneHotEncode().apply(df)
           Born_UK  Born_USA
        1        0         1
        2        1         0
        3        0         0
    """

    class _FitterEncoder(object):
        def __init__(self, col_name, dummy_columns):
            self.col_name = col_name
            self.dummy_columns = dummy_columns

        def __call__(self, value):
            this_dummy = f"{self.col_name}_{value}"
            return pd.Series(
                data=[
                    int(this_dummy == dummy_col)
                    for dummy_col in self.dummy_columns
                ],
                index=self.dummy_columns,
            )

    def __init__(
        self,
        columns=None,
        dummy_na=False,
        exclude_columns=None,
        drop_first=True,
        drop=True,
        **kwargs
    ):
        self._dummy_na = dummy_na
        self._drop_first = drop_first
        self._drop = drop
        self._dummy_col_map = {}
        self._encoder_map = {}
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
            'desc_temp': "One-hot encode {}",
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = OfDtypes(['object', 'category'])
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, df, verbose):
        columns_to_encode = self._get_columns(df, fit=True)
        assign_map = {}
        if verbose:
            columns_to_encode = tqdm.tqdm(columns_to_encode)
        for colname in columns_to_encode:
            if verbose:
                columns_to_encode.set_description(colname)
            dummies = pd.get_dummies(
                df[colname],
                drop_first=False,
                dummy_na=self._dummy_na,
                prefix=colname,
                prefix_sep="_",
            )
            nan_col = colname + "_nan"
            if self._drop_first:
                dfirst_col = colname + "_" + str(self._drop_first)
                if dfirst_col in dummies:
                    if verbose:
                        print(
                            (
                                "Dropping {} dummy column instead of first "
                                "column when one-hot encoding {}."
                            ).format(dfirst_col, colname)
                        )
                    dummies.drop(dfirst_col, axis=1, inplace=True)
                elif nan_col in dummies:
                    dummies.drop(nan_col, axis=1, inplace=True)
                else:
                    dummies.drop(dummies.columns[0], axis=1, inplace=True)
            self._dummy_col_map[colname] = list(dummies.columns)
            self._encoder_map[colname] = OneHotEncode._FitterEncoder(
                colname, list(dummies.columns)
            )
            for column in dummies:
                assign_map[column] = dummies[column]

        inter_df = df.assign(**assign_map)
        self.is_fitted = True
        if self._drop:
            return inter_df.drop(columns_to_encode, axis=1)
        return inter_df

    def _transform(self, df, verbose):
        assign_map = {}
        columns_to_encode = self._get_columns(df, fit=False)
        for colname in columns_to_encode:
            col = df[colname]
            try:
                encoder = self._encoder_map[colname]
            except KeyError:  # pragma: no cover
                raise PipelineApplicationError((
                    "Missing encoder for column {} when applying a fitted "
                    "OneHotEncode pipeline stage by class {} !")
                    .format(colname, self.__class__))
            res_cols = col.apply(encoder)
            for res_col in res_cols:
                assign_map[res_col] = res_cols[res_col]
        inter_df = df.assign(**assign_map)
        if self._drop:
            return inter_df.drop(columns_to_encode, axis=1)
        return inter_df


class ColumnTransformer(ColumnsBasedPipelineStage):
    """A pipeline stage that applies transformation to dataframe columns.

    Parameters
    ----------
    columns : single label, list-like or callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`. If None is provided
        all input columns are transformed.
    result_columns : single label or list-like, default None
        Labels for the new columns resulting from the transformations. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, then the label of the source column is
        used; otherwise, the provided 'suffix' is concatenated to the label of
        the source column.
    drop : bool, default True
        If set to True, source columns are dropped after being transformed.
    suffix : str, default '_transformed'
        The suffix transformed columns gain if no new column labels are given.
    """

    def __init__(
        self,
        columns,
        result_columns=None,
        drop=True,
        suffix=None,
        **kwargs
    ):
        if suffix is None:  # pragma: no cover
            suffix = "_transformed"
        self.suffix = suffix
        self._result_columns = result_columns
        if result_columns:
            self._result_columns = _interpret_columns_param(result_columns)
            if len(self._result_columns) != len(
                    _interpret_columns_param(columns)):
                raise ValueError(
                    "columns and result_columns parameters must"
                    " be label lists of the same length!"
                )
        self._drop = drop
        super_kwargs = {
            'columns': columns,
            'desc_temp': "Transform columns {}",
            'none_columns': 'all',
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    @abc.abstractmethod
    def _col_transform(self, series, label):
        raise NotImplementedError

    def _transformation(self, df, verbose, fit):
        columns = self._get_columns(df, fit=fit)
        result_columns = self._result_columns
        if self._result_columns is None:
            if self._drop:
                result_columns = columns
            else:
                result_columns = [
                    f'{col}{self.suffix}' for col in columns
                ]
        inter_df = df
        for i, colname in enumerate(columns):
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = result_columns[i]
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=self._col_transform(source_col, colname),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


class _AttrGetter():
    """A custom callable that gets a specific attribute from input objects.

    Parameters
    ----------
    attr_name : str
        The name of the attribute to get from input objects.
    """

    def __init__(self, attr_name: str) -> None:
        self.attr_name = attr_name

    def __call__(self, obj: object) -> object:
        return getattr(obj, self.attr_name)


class _MethodRetValGetter():
    """A custom callable that gets the return value of a specified method with
    specified keyword arguments from input objects.

    Parameters
    ----------
    method_name : str
        The name of the method to call for input objects.
    method_kwargs : dict of str to object
        The keyword arguments to supply to the specified method on each call.
    """

    def __init__(
        self,
        method_name: str,
        method_kwargs: Dict[str, object],
    ) -> None:
        self.method_name = method_name
        self.method_kwargs = method_kwargs

    def __call__(self, obj: object) -> object:
        return getattr(obj, self.method_name)(**self.method_kwargs)


class MapColVals(ColumnTransformer):
    """A pipeline stage that replaces the values of a column by a map.

    Parameters
    ----------
    columns : single label, list-like or callable
        Column labels in the DataFrame to be mapped. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`. If None is provided
        all input columns are mapped.
    value_map : dict, pandas.Series, callable, str or tuple
        The value-to-value map to use, mapping existing values to new one. If a
        dictionary is provided, its mapping is used. Values not in the
        dictionary as keys will be converted to NaN. If a Series is given,
        values are mapped by its index to its values. If a callable is given,
        it is applied element-wise to given columns. If a string is given, it
        is interpreted as the name of an attribute or a property of the series
        values to use as target values. If a tuple is provided, its first
        element is expected to be a string, interpreted as a name of a method
        of the series values to call, and its second element is expected to be
        a dict - possibly empty - mapping additional keyword arguments names
        to their values.
    result_columns : single label or list-like, default None
        Labels for the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, then the label of the source column is
        used; otherwise, the label of the source column is used with the suffix
        given ("_map" by default).
    drop : bool, default True
        If set to True, source columns are dropped after being mapped.
    suffix : str, default '_map'
        The suffix mapped columns gain if no new column labels are given.

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

        >>> from datetime import timedelta;
        >>> df = pd.DataFrame(
        ...    data=[
        ...       [timedelta(weeks=2)],
        ...       [timedelta(weeks=4)],
        ...       [timedelta(weeks=10)]
        ...    ],
        ...    index=['proposal', 'midterm', 'finals'],
        ...    columns=['Due'],
        ... )
        >>> pdp.MapColVals('Due', ('total_seconds', {})).apply(df)
                        Due
        proposal  1209600.0
        midterm   2419200.0
        finals    6048000.0
    """

    def __init__(
        self,
        columns: ColumnsParamType,
        value_map: Union[dict, pd.Series, Callable, str, Tuple[str, dict]],
        result_columns: Optional[ColumnLabelsType] = None,
        drop: Optional[bool] = True,
        suffix: Optional[str] = None,
        **kwargs: Dict[str, object],
    ):
        self._value_map = value_map
        self._applied_value_map = value_map
        if type(value_map) == str:
            self._applied_value_map = _AttrGetter(attr_name=value_map)
        elif type(value_map) == tuple:
            self._applied_value_map = _MethodRetValGetter(
                method_name=value_map[0],
                method_kwargs=value_map[1],
            )
        if suffix is None:
            suffix = "_map"
        _, colstr = ColumnsBasedPipelineStage._interpret_columns_param(
            columns)
        super_kwargs = {
            'columns': columns,
            'result_columns': result_columns,
            'drop': drop,
            'suffix': suffix,
            'desc': f"Map values of columns {colstr} with {self._value_map}.",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _col_transform(self, series, label):
        return series.map(self._applied_value_map)


def _always_true(x):
    return True


class ApplyToRows(PdPipelineStage):
    """A pipeline stage generating columns by applying a function to each row.

    Parameters
    ----------
    func : function
        The function to be applied to each row of the processed DataFrame.
    colname : single label, default None
        The label of the new column resulting from the function application. If
        None, 'new_col' is used. Ignored if a DataFrame is generated by the
        function (i.e. each row generates a Series rather than a value), in
        which case the label of each column in the resulting DataFrame is used.
    follow_column : str, default None
        Resulting columns will be inserted after this column. If None, new
        columns are inserted at the end of the processed DataFrame.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.
    prec : function, default None
        A function taking a DataFrame, returning True if this stage is
        applicable to the given DataFrame. If None is given, a function always
        returning True is used.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3, 2143], [10, 1321], [7, 1255]]
        >>> df = pd.DataFrame(data, [1,2,3], ['years', 'avg_revenue'])
        >>> total_rev = lambda row: row['years'] * row['avg_revenue']
        >>> add_total_rev = pdp.ApplyToRows(total_rev, 'total_revenue')
        >>> add_total_rev(df)
           years  avg_revenue  total_revenue
        1      3         2143           6429
        2     10         1321          13210
        3      7         1255           8785

        >>> def halfer(row):
        ...     new = {'year/2': row['years']/2, 'rev/2': row['avg_revenue']/2}
        ...     return pd.Series(new)
        >>> half_cols = pdp.ApplyToRows(halfer, follow_column='years')
        >>> half_cols(df)
           years   rev/2  year/2  avg_revenue
        1      3  1071.5     1.5         2143
        2     10   660.5     5.0         1321
        3      7   627.5     3.5         1255
    """

    _DEF_APPLYTOROWS_EXC_MSG = "Applying a function {} failed."
    _DEF_COLNAME = "new_col"

    def __init__(
        self,
        func,
        colname=None,
        follow_column=None,
        func_desc=None,
        prec=None,
        **kwargs
    ):
        if colname is None:
            colname = ApplyToRows._DEF_COLNAME
        if func_desc is None:
            func_desc = ""
        if prec is None:
            prec = _always_true
        self._func = func
        self._colname = colname
        self._follow_column = follow_column
        self._func_desc = func_desc
        self._prec_func = prec
        super_kwargs = {
            "exmsg": ApplyToRows._DEF_APPLYTOROWS_EXC_MSG.format(func_desc),
            "desc": f"Generating a column with a function {self._func_desc}.",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._prec_func(df)

    def _transform(self, df, verbose):
        new_cols = df.apply(self._func, axis=1)
        if isinstance(new_cols, pd.Series):
            loc = len(df.columns)
            if self._follow_column:
                loc = df.columns.get_loc(self._follow_column) + 1
            return out_of_place_col_insert(
                df=df, series=new_cols, loc=loc, column_name=self._colname
            )
        if isinstance(new_cols, pd.DataFrame):
            sorted_cols = sorted(list(new_cols.columns))
            new_cols = new_cols[sorted_cols]
            if self._follow_column:
                inter_df = df
                loc = df.columns.get_loc(self._follow_column) + 1
                for colname in new_cols.columns:
                    inter_df = out_of_place_col_insert(
                        df=inter_df,
                        series=new_cols[colname],
                        loc=loc,
                        column_name=colname,
                    )
                    loc += 1
                return inter_df
            assign_map = {
                colname: new_cols[colname] for colname in new_cols.columns
            }
            return df.assign(**assign_map)
        raise TypeError(  # pragma: no cover
            "Unexpected type generated by applying a function to a DataFrame."
            " Only Series and DataFrame are allowed."
        )


class ApplyByCols(ColumnTransformer):
    """A pipeline stage applying an element-wise function to columns.
    For applying series-wise function, see `AggByCols`.

    Parameters
    ----------
    columns : single label, list-like or callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    func : function
        The function to be applied to each element of the given columns.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the name of the source column is used;
        otherwise, the name of the source column is used with the suffix
        '_app'.
    drop : bool, default True
        If set to True, source columns are dropped after being mapped.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. Optional.
    suffix : str, default None
        If provided, this string is concated to resulting column labels instead
        of '_app'.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import math;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> round_ph = pdp.ApplyByCols("ph", math.ceil)
        >>> round_ph(df)
           ph  lbl
        1   4  acd
        2   8  alk
        3  13  alk
    """

    def __init__(
        self,
        columns,
        func,
        result_columns=None,
        drop=True,
        func_desc=None,
        suffix=None,
        **kwargs
    ):
        self._func = func
        if suffix is None:
            suffix = "_app"
        if func_desc is None:
            func_desc = ""
        self._func_desc = func_desc
        super_kwargs = {
            'columns': columns,
            'result_columns': result_columns,
            'drop': drop,
            'suffix': suffix,
            'desc_temp': f'Apply a function {func_desc} to columns {{}}',
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _col_transform(self, series, label):
        return series.apply(self._func)


class ColByFrameFunc(PdPipelineStage):
    """A pipeline stage adding a column by applying a dataframe-wide function.

    Note that assigning `column` with the label of an existing column and
    providing the same label to the `before_column` parameter will result in
    replacing the original column at the same location.

    Parameters
    ----------
    column : str
        The label of the resulting column. If its the label of an existing
        column it will replace that column.
    func : function
        The function to be applied to the input dataframe. The function should
        return a pandas.Series object.
    follow_column : str, default None
        Resulting columns will be inserted after this column. If both this
        parameter and `before_column` are None, new columns are inserted at the
        end of the processed DataFrame.
    before_column : str, default None
        Resulting columns will be inserted before this column. If both this
        parameter and `follow_colum` are None, new columns are inserted at the
        end of the processed DataFrame. If both are provided, `before_column`
        takes precedence.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3, 3], [2, 4], [1, 5]]
        >>> df = pd.DataFrame(data, [1,2,3], ["A","B"])
        >>> func = lambda df: df['A'] == df['B']
        >>> add_equal = pdp.ColByFrameFunc("A==B", func)
        >>> add_equal(df)
           A  B   A==B
        1  3  3   True
        2  2  4  False
        3  1  5  False
    """

    _BASE_STR = "Applying a function{} to generate column {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_DESCRIPTION_SUFFIX = "."

    def __init__(
        self, column, func, follow_column=None, before_column=None,
        func_desc=None, **kwargs
    ):
        self._column = column
        self._func = func
        self._follow_column = follow_column
        self._before_column = before_column
        if func_desc is None:
            func_desc = ""
        else:
            func_desc = " " + func_desc
        self._func_desc = func_desc
        base_str = ColByFrameFunc._BASE_STR.format(self._func_desc, column)
        super_kwargs = {
            "exmsg": base_str + ColByFrameFunc._DEF_EXC_MSG_SUFFIX,
            "desc": base_str + ColByFrameFunc._DEF_DESCRIPTION_SUFFIX,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        inter_df = df
        try:
            new_col = self._func(df)
        except Exception:
            raise PipelineApplicationError(
                f"Exception raised applying function {self._func_desc} to "
                f"dataframe by class {self.__class__}."
            )
        if self._follow_column:
            loc = df.columns.get_loc(self._follow_column) + 1
        elif self._before_column:
            loc = df.columns.get_loc(self._before_column)
        else:
            loc = len(df.columns)
        inter_df = out_of_place_col_insert(
            df=inter_df, series=new_col, loc=loc, column_name=self._column
        )
        return inter_df


class AggByCols(ColumnTransformer):
    """A pipeline stage applying a series-wise function to columns.
    For applying element-wise function, see `ApplyByCols`.

    Parameters
    ----------
    columns : single label, list-like or callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    func : function
        The function to be applied to each of the given columns. Must work when
        given a pandas.Series object and return either a Scaler or
        `pandas.Series``. If a Scaler is returned, the result is broadcasted
        into a column of the original length.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the name of the source column is used;
        otherwise, the name of the source column is used with a defined suffix.
    drop : bool, default True
        If set to True, source columns are dropped after being mapped.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.
    suffix : str, optional
        The suffix to add to resulting columns in case where results_columns
        is None and drop is set to False. Of not given, defaults to '_agg'.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import numpy as np;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> log_ph = pdp.AggByCols("ph", np.log)
        >>> log_ph(df)
                 ph  lbl
        1  1.163151  acd
        2  1.974081  alk
        3  2.493205  alk

        >>> min_ph = pdp.AggByCols("ph", min, drop=False, suffix='_min')
        >>> min_ph(df)
             ph  ph_min  lbl
        1   3.2     3.2  acd
        2   7.2     3.2  alk
        3  12.1     3.2  alk
    """

    def __init__(
        self,
        columns,
        func,
        result_columns=None,
        drop=True,
        func_desc=None,
        suffix=None,
        **kwargs
    ):
        self._func = func
        if suffix is None:
            suffix = "_agg"
        if func_desc is None:
            func_desc = ""
        self._func_desc = func_desc
        super_kwargs = {
            'columns': columns,
            'result_columns': result_columns,
            'drop': drop,
            'suffix': suffix,
            'desc_temp': (
                f'Apply an aggregation function {func_desc} to columns {{}}'
            ),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _col_transform(self, series, label):
        return series.agg(self._func)


class Log(ColumnsBasedPipelineStage):
    """A pipeline stage that log-transforms numeric data.

    Parameters
    ----------
    columns : single label, list-like or callable, default None
        Column names in the DataFrame to be encoded. If columns is None then
        all the columns with a numeric dtype will be transformed, except those
        given in the exclude_columns parameter. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`.
    exclude_columns : single label, list-like or callable, default None
        Label or labels of columns to be excluded from encoding. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
        pandas.DataFrame. See `pdpipe.cq`. Optional.
    drop : bool, default False
        If set to True, the source columns are dropped after being encoded,
        and the resulting encoded columns retain the names of the source
        columns. Otherwise, encoded columns gain the suffix '_log'.
    non_neg : bool, default False
        If True, each transformed column is first shifted by the smallest
        negative value it includes (non-negative columns are thus not shifted).
    const_shift : int, optional
        If given, each transformed column is first shifted by this constant. If
        non_neg is True then that transformation is applied first, and only
        then is the column shifted by this constant.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> log_stage = pdp.Log("ph", drop=True)
        >>> log_stage(df)
                 ph  lbl
        1  1.163151  acd
        2  1.974081  alk
        3  2.493205  alk
    """

    _DEF_LOG_APP_MSG = "Log-transforming {}..."

    def __init__(
        self,
        columns=None,
        exclude_columns=None,
        drop=False,
        non_neg=False,
        const_shift=None,
        **kwargs
    ):
        self._drop = drop
        self._non_neg = non_neg
        self._const_shift = const_shift
        self._col_to_minval = {}
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
            'desc_temp': "Log-transform {}",
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = OfDtypes([np.number])
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, df, verbose):
        columns_to_transform = self._get_columns(df, fit=True)
        if verbose:
            columns_to_transform = tqdm.tqdm(columns_to_transform)
        inter_df = df
        for colname in columns_to_transform:
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_log"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            new_col = source_col
            if self._non_neg:
                minval = min(new_col)
                if minval < 0:
                    new_col = new_col + abs(minval)
                    self._col_to_minval[colname] = abs(minval)
                else:
                    self._col_to_minval[colname] = 0
            # must check not None as neg numbers eval to False
            if self._const_shift is not None:
                new_col = new_col + self._const_shift
            new_col = np.log(new_col)
            inter_df = out_of_place_col_insert(
                df=inter_df, series=new_col, loc=loc, column_name=new_name
            )
        self.is_fitted = True
        return inter_df

    def _transform(self, df, verbose):
        inter_df = df
        columns_to_transform = self._get_columns(df, fit=False)
        if verbose:
            columns_to_transform = tqdm.tqdm(columns_to_transform)
        for colname in columns_to_transform:
            try:
                source_col = df[colname]
            except KeyError:  # pragma: no cover
                raise PipelineApplicationError((
                    "Missig column {} when applying a fitted "
                    "Log pipeline stage by class {} !").format(
                        colname, self.__class__))
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_log"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            new_col = source_col
            if self._non_neg:
                if colname in self._col_to_minval:
                    absminval = self._col_to_minval[colname]
                    new_col = new_col + absminval
                else:  # pragma: no cover
                    raise PipelineApplicationError((
                        "Missig fitted parameter for column {} when applying a"
                        " fitted Log pipeline stage by class {}!").format(
                            colname, self.__class__))
            # must check not None as neg numbers eval to False
            if self._const_shift is not None:
                new_col = new_col + self._const_shift
            new_col = np.log(new_col)
            inter_df = out_of_place_col_insert(
                df=inter_df, series=new_col, loc=loc, column_name=new_name
            )
        return inter_df
