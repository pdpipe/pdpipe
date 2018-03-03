"""Pipeline stages dependent on the scikit-learn Python library."""

import pandas as pd
import sklearn.preprocessing
import tqdm
from skutil.preprocessing import scaler_by_params

from pdpipe.core import PipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.shared import (
    _interpret_columns_param,
    _list_str
)


class Encode(PipelineStage):
    """A pipeline stage that encodes categorical columns to integer values.

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
        columns. Otherwise, encoded columns gain the suffix '_enc'.

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
        self.encoders = {}
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
        self.is_fitted = True
        return inter_df

    def _transform(self, df, verbose):
        inter_df = df
        for colname in self.encoders:
            lbl_enc = self.encoders[colname]
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_enc"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=lbl_enc.transform(source_col),
                loc=loc,
                column_name=new_name)
        return inter_df


class Scale(PipelineStage):
    """A pipeline stage that scales data.

    Parameters
    ----------
    scaler : str
        The type of scaler to use to scale the data. One of 'StandardScaler',
        'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer'
        and 'Normalizer'.
    exclude_columns : str or list-like, default None
        Name or names of columns to be excluded from scaling. Excluded columns
        are appended to the end of the resulring dataframe. If set to None, all
        columns of dtype object are excluded from scaling.
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the scaler
        constructor on scaler creation (e.g. 'n_quantiles' for
        QuantileTransformer). PipelineStage valid keyword arguments are used
        to override Scale class defaults.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, 0.3], [7.2, 0.35], [12.1, 0.29]]
    >>> df = pd.DataFrame(data, [1,2,3], ["ph","gt"])
    >>> scale_stage = pdp.Scale("StandardScaler")
    >>> scale_stage(df)
             ph        gt
    1 -1.181449 -0.508001
    2 -0.082427  1.397001
    3  1.263876 -0.889001
    """

    _DESC_PREFIX = "Scale data"
    _DEF_SCALE_EXC_MSG = "Scale stage failed."
    _DEF_SCALE_APP_MSG = "Scaling data..."

    def __init__(self, scaler, exclude_columns=None, **kwargs):
        self.scaler = scaler
        if exclude_columns is None:
            self._exclude_columns = None
            desc_suffix = "."
        else:
            self._exclude_columns = _interpret_columns_param(
                exclude_columns, 'exclude_columns')
            col_str = _list_str(self._exclude_columns)
            desc_suffix = " except columns {}.".format(col_str)
        super_kwargs = {
            'exmsg': Scale._DEF_SCALE_EXC_MSG,
            'appmsg': Scale._DEF_SCALE_APP_MSG,
            'desc': Scale._DESC_PREFIX + desc_suffix,
        }
        self._kwargs = kwargs
        valid_super_kwargs = super()._init_kwargs()
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _op(self, df, verbose):
        if self._exclude_columns is None:
            self._exclude_columns = list(
                (df.dtypes[df.dtypes == object]).index)
        self._col_order = list(df.columns)
        if len(self._exclude_columns) > 0:
            exclude = df[self._exclude_columns]
            apply_to = df[[
                col for col in df.columns if col not in self._exclude_columns]]
        else:
            apply_to = df
        self._scaler = scaler_by_params(self.scaler, **self._kwargs)
        res = pd.DataFrame(
            data=self._scaler.fit_transform(apply_to),
            index=apply_to.index,
            columns=apply_to.columns,
        )
        if len(self._exclude_columns) > 0:
            res = pd.concat([res, exclude], axis=1)
            res = res[self._col_order]
        self.is_fitted = True
        return res

    def _transform(self, df, verbose):
        if len(self._exclude_columns) > 0:
            exclude = df[self._exclude_columns]
            apply_to = df[[
                col for col in df.columns if col not in self._exclude_columns]]
        else:
            apply_to = df
        res = pd.DataFrame(
            data=self._scaler.transform(apply_to),
            index=apply_to.index,
            columns=apply_to.columns,
        )
        if len(self._exclude_columns) > 0:
            res = pd.concat([res, exclude], axis=1)
            res = res[self._col_order]
        return res
