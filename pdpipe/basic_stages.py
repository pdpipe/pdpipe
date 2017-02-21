"""Basic pdpipe PiplineStages."""

import pandas as pd
import sortedcontainers as sc
import sklearn.preprocessing
import tqdm

from .core import PipelineStage


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


class ColDrop(PipelineStage):
    """A pipline stage that drops columns by name.

    Parameters
    ----------
    columns : str or iterable
        The name, or an iterable of names, of columns to drop.
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Defaults to True.


    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> pdp.ColDrop('num').apply(df)
      char
    1    a
    2    b
    """

    DEF_COLDROP_EXC_MSG = "ColDrop stage failed because not all columns {}"\
                          " were found in input dataframe."
    DEF_COLDROP_APPLY_MSG = 'Dropping columns {}...'

    def __init__(self, columns, exraise=True):
        self._columns = _interpret_columns_param(columns, 'columns')
        super(ColDrop, self).__init__(
            exraise=exraise,
            exmsg=ColDrop.DEF_COLDROP_EXC_MSG.format(self._columns),
            appmsg=ColDrop.DEF_COLDROP_APPLY_MSG.format(self._columns)
        )

    def _prec(self, df):
        return set(self._columns).issubset(df.columns)

    def _op(self, df, verbose):
        return df.drop(self._columns, axis=1)

    def __str__(self):
        return "Drop columns {}".format(self._columns)


class ValDrop(PipelineStage):
    """A pipline stage that drops rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to drop.
    columns : str or list-like, defualt None
        The name, or an iterable of names, of columns to check for the given
        values. If set to None, all columns are checked.
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Defaults to True.


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

    DEF_VALDROP_EXC_MSG = "ValDrop stage failed because not all columns {}"\
                          " were found in input dataframe."
    DEF_VALDROP_APPLY_MSG = "Dropping values {}..."

    def __init__(self, values, columns=None, exraise=True):
        self._values = values
        if columns is None:
            self._columns = None
            apply_msg = ValDrop.DEF_VALDROP_APPLY_MSG.format(self._values)
        else:
            self._columns = _interpret_columns_param(columns, 'columns')
            apply_msg = ValDrop.DEF_VALDROP_APPLY_MSG.format(
                "{} in {}".format(self._values, self._columns))
        super(ValDrop, self).__init__(
            exraise=exraise,
            exmsg=ValDrop.DEF_VALDROP_EXC_MSG.format(self._columns),
            appmsg=apply_msg
        )

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

    def __str__(self):
        if self._columns:
            return "Drop values {} in columns {}".format(
                self._values, self._columns)
        return "Drop values {}".format(self._values)


class ValKeep(PipelineStage):
    """A pipline stage that keeps rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to keep.
    columns : str or list-like, defualt None
        The name, or an iterable of names, of columns to check for the given
        values. If set to None, all columns are checked.
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Defaults to True.


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

    DEF_VALKEEP_EXC_MSG = "ValKeep stage failed because not all columns {}"\
                          " were found in input dataframe."
    DEF_VALKEEP_APPLY_MSG = "Keeping values {}..."

    def __init__(self, values, columns=None, exraise=True):
        self._values = values
        if columns is None:
            self._columns = None
            apply_msg = ValKeep.DEF_VALKEEP_APPLY_MSG.format(self._values)
        else:
            self._columns = _interpret_columns_param(columns, 'columns')
            apply_msg = ValKeep.DEF_VALKEEP_APPLY_MSG.format(
                "{} in {}".format(self._values, self._columns))
        super(ValKeep, self).__init__(
            exraise=exraise,
            exmsg=ValKeep.DEF_VALKEEP_EXC_MSG.format(self._columns),
            appmsg=apply_msg
        )

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

    def __str__(self):
        if self._columns:
            return "Keep values {} in columns {}".format(
                self._values, self._columns)
        return "Keep values {}".format(self._values)


class ColRename(PipelineStage):
    """A pipline stage that renames a column or columns.

    Parameters
    ----------
    rename_map : dict
        Maps old column names to new ones.
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Defaults to True.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> pdp.ColRename({'num': 'len', 'char': 'initial'}).apply(df)
       len initial
    1    8       a
    2    5       b
    """

    DEF_COLDRENAME_EXC_MSG = "ColRename stage failed because not all columns "\
                          "{} were found in input dataframe."
    DEF_COLDRENAME_APP_MSG = "Renaming columns {}..."

    def __init__(self, rename_map, exraise=True):
        super(ColRename, self).__init__(
            exraise=exraise,
            exmsg=ColRename.DEF_COLDRENAME_EXC_MSG.format(
                list(rename_map.keys())),
            appmsg=ColRename.DEF_COLDRENAME_APP_MSG.format(
                list(rename_map.keys()))
        )
        self._rename_map = rename_map

    def _prec(self, df):
        return set(self._rename_map.keys()).issubset(df.columns)

    def _op(self, df, verbose):
        return df.rename(columns=self._rename_map)

    def __str__(self):
        return "Rename columns {}".format(self._rename_map)


class Bin(PipelineStage):
    """A pipline stage that adds a binned version of a column or columns.

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
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[-3],[4],[5], [9]], [1,2,3, 4], ['speed'])
    >>> pdp.Bin({'speed': [5]}, drop=False).apply(df)
       speed speed_binned
    1     -3          < 5
    2      4          < 5
    3      5          5 ≤
    4      9          5 ≤
    >>> pdp.Bin({'speed': [0,5,8]}, drop=False).apply(df)
       speed speed_binned
    1     -3          < 0
    2      4          0-5
    3      5          5-8
    4      9          8 ≤
    """

    DEF_BIN_EXC_MSG = "Bin stage failed because not all columns "\
                          "{} were found in input dataframe."
    DEF_BIN_APP_MSG = "Binning columns {}..."

    def __init__(self, bin_map, drop=True, exraise=True):
        super(Bin, self).__init__(
            exraise=exraise,
            exmsg=Bin.DEF_BIN_EXC_MSG.format(
                list(bin_map.keys())),
            appmsg=Bin.DEF_BIN_APP_MSG.format(
                list(bin_map.keys()))
        )
        self._bin_map = bin_map
        self._drop = drop

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
                    return '{} ≤'.format(sorted_bins[-1])
                return '{}-{}'.format(sorted_bins[ind], sorted_bins[ind+1])
            try:
                ind = sorted_bins.bisect(val)
                if ind == 0:
                    return '< {}'.format(sorted_bins[ind])
                return '{}-{}'.format(sorted_bins[ind-1], sorted_bins[ind])
            except IndexError:
                return '{} ≤'.format(sorted_bins[sorted_bins.bisect(val)-1])
        return _col_binner

    def _op(self, df, verbose):
        assign_map = {}
        colnames = list(self._bin_map.keys())
        if verbose:
            colnames = tqdm.tqdm(colnames)
        for colname in colnames:
            assign_map[colname+'_binned'] = df[colname].apply(
                self._get_col_binner(self._bin_map[colname]))
        inter_df = df.assign(**assign_map)
        if self._drop:
            return inter_df.drop(list(self._bin_map.keys()), axis=1)
        return inter_df

    def __str__(self):
        string = ""
        for col in self._bin_map:
            string += "Bin {} by {}, ".format(col, self._bin_map[col])
        string = string[0:-2] + '.'
        return string


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
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([['USA'], ['UK'], ['Greece']], [1,2,3], ['Born'])
    >>> pdp.Binarize().apply(df)
       Born.UK  Born.USA
    1      0.0       1.0
    2      1.0       0.0
    3      0.0       0.0
    """

    DEF_BINAR_EXC_MSG = "Binarize stage failed because not all columns "\
                          "{} were found in input dataframe."
    DEF_BINAR_APP_MSG = "Binarizing {}..."

    def __init__(self, columns=None, dummy_na=False, exclude_columns=None,
                 drop_first=True, drop=True, exraise=True):
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
        super(Binarize, self).__init__(
            exraise=exraise,
            exmsg=Binarize.DEF_BINAR_EXC_MSG.format(self._columns),
            appmsg=Binarize.DEF_BINAR_APP_MSG.format(
                self._columns or "all columns")
        )

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
            dummis = pd.get_dummies(
                df[colname], drop_first=self._drop_first,
                dummy_na=self._dummy_na, prefix=colname, prefix_sep='.')
            for column in dummis:
                assign_map[column] = dummis[column]
        inter_df = df.assign(**assign_map)
        if self._drop:
            return inter_df.drop(columns_to_binar, axis=1)
        return inter_df

    def __str__(self):
        return "Binarize {}".format(self._columns or "all categorical columns")


class MapColVals(PipelineStage):
    """A pipline stage that replaces the values of a column by a map.

    Parameters
    ----------
    column_name : str
        The name of the column to apply the value map for.
    value_map : dict
        A dictionary mapping existing values to new ones. Not all existing
        values need to be referenced; missing one will neither be changed nor
        dropped.
    result_col_name : str, default None
        The name of the new column resulting from the mapping operation. If
        None, behavior depends on the drop parameter: If drop is True, the
        name of the source column is used; otherwise, the name of the source
        column is used with the suffix '_map'.
    drop : bool, default True
        If set to True, the source column is dropped after being mapped.
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped.

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

    DEF_MAP_COL_VAL_EXC_MSG = "MapColVals stage failed because column "\
                          "{} was not found in input dataframe."
    DEF_MAP_COL_VAL_APP_MSG = "Mapping values of column {}..."

    def __init__(self, column_name, value_map, result_col_name=None,
                 drop=True, exraise=True):
        self._column_name = column_name
        self._value_map = value_map
        self._result_col_name = result_col_name
        if result_col_name is None:
            if drop:
                self._result_col_name = self._column_name
            else:
                self._result_col_name = self._column_name + '_map'
        self._drop = drop
        super(MapColVals, self).__init__(
            exraise=exraise,
            exmsg=MapColVals.DEF_MAP_COL_VAL_EXC_MSG.format(self._column_name),
            appmsg=MapColVals.DEF_MAP_COL_VAL_APP_MSG.format(self._column_name)
        )

    def _prec(self, df):
        return self._column_name in df.columns

    def _op(self, df, verbose):
        inter_df = df
        source_column = df[self._column_name]
        if self._drop:
            inter_df = df.drop(self._column_name, axis=1)
        inter_df = df.assign(**{
            self._result_col_name: source_column.replace(self._value_map)
        })
        return inter_df

    def __str__(self):
        return "Map values of column {}.".format(self._column_name)


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
        If set to True, the source columns are dropped after being encoded.
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
    >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
    >>> encode_stage = pdp.Encode("lbl")
    >>> encode_stage(df)
         ph  lbl_enc
    1   3.2        0
    2   7.2        1
    3  12.1        1
    >>> encode_stage.encoders["lbl"].inverse_transform([0,1,1])
    array(['acd', 'alk', 'alk'], dtype=object)
    """

    DEF_ENCODE_EXC_MSG = "Encode stage failed because not all columns "\
                          "{} were found in input dataframe."
    DEF_ENCODE_APP_MSG = "Encoding {}..."

    def __init__(self, columns=None, exclude_columns=None, drop=True,
                 exraise=True):
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
        super(Encode, self).__init__(
            exraise=exraise,
            exmsg=Encode.DEF_ENCODE_EXC_MSG.format(self._columns),
            appmsg=Encode.DEF_ENCODE_APP_MSG.format(self._columns)
        )

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _op(self, df, verbose):
        columns_to_encode = self._columns
        if self._columns is None:
            columns_to_encode = list(set(df.select_dtypes(
                include=['object', 'category']).columns).difference(
                    self._exclude_columns))
        assign_map = {}
        if verbose:
            columns_to_encode = tqdm.tqdm(columns_to_encode)
        for colname in columns_to_encode:
            lbl_enc = sklearn.preprocessing.LabelEncoder()
            assign_map[colname+'_enc'] = lbl_enc.fit_transform(
                df[colname])
            self.encoders[colname] = lbl_enc
        inter_df = df.assign(**assign_map)
        if self._drop:
            return inter_df.drop(columns_to_encode, axis=1)
        return inter_df

    def __str__(self):
        return "Encode {}".format(self._columns or "all categorical columns")
