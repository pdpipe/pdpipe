"""Basic pdpipe PiplineStages."""

import pandas as pd
import sortedcontainers as sc
from tqdm import tqdm

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
    exraise : bool, optional
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
        self.columns = _interpret_columns_param(columns, 'columns')
        super(ColDrop, self).__init__(
            exraise=exraise,
            exmsg=ColDrop.DEF_COLDROP_EXC_MSG.format(self.columns),
            appmsg=ColDrop.DEF_COLDROP_APPLY_MSG.format(self.columns)
        )

    def _prec(self, df):
        return set(self.columns).issubset(df.columns)

    def _op(self, df, verbose):
        return df.drop(self.columns, axis=1)

    def __str__(self):
        return "Drop columns {}".format(self.columns)


class ColRename(PipelineStage):
    """A pipline stage that renames a column or columns.

    Parameters
    ----------
    rename_map : dict
        Maps old column names to new ones.
    exraise : bool, optional
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
        self.rename_map = rename_map

    def _prec(self, df):
        return set(self.rename_map.keys()).issubset(df.columns)

    def _op(self, df, verbose):
        return df.rename(columns=self.rename_map)

    def __str__(self):
        return "Rename columns {}".format(self.rename_map)


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
        self.bin_map = bin_map
        self.drop = drop

    def _prec(self, df):
        return set(self.bin_map.keys()).issubset(df.columns)

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
        colnames = list(self.bin_map.keys())
        if verbose:
            colnames = tqdm(colnames)
        for colname in colnames:
            assign_map[colname+'_binned'] = df[colname].apply(
                self._get_col_binner(self.bin_map[colname]))
        inter_df = df.assign(**assign_map)
        if self.drop:
            return inter_df.drop(list(self.bin_map.keys()), axis=1)
        return inter_df

    def __str__(self):
        string = ""
        for col in self.bin_map:
            string += "Bin {} by {}, ".format(col, self.bin_map[col])
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
    drop_first : bool, default True
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    drop : bool, default True
        If set to True, the source columns are dropped after being binarized.
    exraise : bool, optional
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Defaults to True.

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
    DEF_BINAR_APP_MSG = "Binarize columns {}..."

    def __init__(self, columns=None, dummy_na=False, exclude_columns=None,
                 drop_first=True, drop=True, exraise=True):
        if columns is None:
            self.columns = None
        else:
            self.columns = _interpret_columns_param(columns, 'columns')
        self.dummy_na = dummy_na
        if exclude_columns is None:
            self.exclude_columns = []
        else:
            self.exclude_columns = _interpret_columns_param(
                exclude_columns, 'exclude_columns')
        self.drop_first = drop_first
        self.drop = drop
        super(Binarize, self).__init__(
            exraise=exraise,
            exmsg=Binarize.DEF_BINAR_EXC_MSG.format(self.columns),
            appmsg=Binarize.DEF_BINAR_APP_MSG.format(self.columns)
        )

    def _prec(self, df):
        return set(self.columns or []).issubset(df.columns)

    def _op(self, df, verbose):
        columns_to_encode = self.columns
        if self.columns is None:
            columns_to_encode = list(set(df.select_dtypes(
                include=['object', 'category']).columns).difference(
                    self.exclude_columns))
        assign_map = {}
        if verbose:
            columns_to_encode = tqdm(columns_to_encode)
        for colname in columns_to_encode:
            dummis = pd.get_dummies(
                df[colname], drop_first=self.drop_first,
                dummy_na=self.dummy_na, prefix=colname, prefix_sep='.')
            for column in dummis:
            #     if '_' in colname:
            #         colname = colname[0:colname.find('_')]
            #     try:
            #         new_colname = '{}.{}'.format(colname, int(col))
            #     except ValueError:
            #         new_colname = '{}.{}'.format(colname, col)
                assign_map[column] = dummis[column]
        inter_df = df.assign(**assign_map)
        if self.drop:
            return inter_df.drop(columns_to_encode, axis=1)
        return inter_df

    def __str__(self):
        if self.columns is None:
            return "Binarize all categorical columns"
        return "Binarize columns {}".format(self.columns)
