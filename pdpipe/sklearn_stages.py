"""PdPipeline stages dependent on the scikit-learn Python library.

Please note that the scikit-learn Python package must be installed for the
stages in this module to work.

When attempting to load stages from this module, pdpipe will first attempt to
import sklearn. If it fails, it will issue a warning, will not import any of
the pipeline stages that make up this module, and continue to load other
pipeline stages.
"""

import pandas as pd
import sklearn.preprocessing
import tqdm
from skutil.preprocessing import scaler_by_params
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from pdpipe.core import PdPipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.shared import (
    _interpret_columns_param,
    _list_str,
    _get_args_list,
)

from .exceptions import PipelineApplicationError


class Encode(PdPipelineStage):
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

    _DEF_ENCODE_EXC_MSG = (
        "Encode stage failed because not all columns "
        "{} were found in input dataframe."
    )
    _DEF_ENCODE_APP_MSG = "Encoding {}..."

    def __init__(
        self, columns=None, exclude_columns=None, drop=True, **kwargs
    ):
        if columns is None:
            self._columns = None
        else:
            self._columns = _interpret_columns_param(columns)
        if exclude_columns is None:
            self._exclude_columns = []
        else:
            self._exclude_columns = _interpret_columns_param(exclude_columns)
        self._drop = drop
        self.encoders = {}
        col_str = _list_str(self._columns)
        super_kwargs = {
            "exmsg": Encode._DEF_ENCODE_EXC_MSG.format(col_str),
            "appmsg": Encode._DEF_ENCODE_APP_MSG.format(col_str),
            "desc": "Encode {}".format(col_str or "all categorical columns"),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _fit_transform(self, df, verbose):
        self.encoders = {}
        columns_to_encode = self._columns
        if self._columns is None:
            columns_to_encode = list(
                set(
                    df.select_dtypes(include=["object", "category"]).columns
                ).difference(self._exclude_columns)
            )
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
                column_name=new_name,
            )
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
                column_name=new_name,
            )
        return inter_df


class Scale(PdPipelineStage):
    """A pipeline stage that scales data.

    Parameters
    ----------
    scaler : str
        The type of scaler to use to scale the data. One of 'StandardScaler',
        'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer'
        and 'Normalizer'.
    exclude_columns : str or list-like, default None
        Name or names of columns to be excluded from scaling. Excluded columns
        are appended to the end of the resulting dataframe.
    exclude_object_columns : bool, default True
        If set to True, all columns of dtype object are added to the list of
        columns excluded from scaling.
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the scaler
        constructor on scaler creation (e.g. 'n_quantiles' for
        QuantileTransformer). PdPipelineStage valid keyword arguments are used
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

    def __init__(
        self,
        scaler,
        exclude_columns=None,
        exclude_object_columns=True,
        **kwargs
    ):
        self.scaler = scaler
        if exclude_columns is None:
            self._exclude_columns = []
            desc_suffix = "."
        else:
            self._exclude_columns = _interpret_columns_param(exclude_columns)
            col_str = _list_str(self._exclude_columns)
            desc_suffix = " except columns {}.".format(col_str)
        self._exclude_obj_cols = exclude_object_columns
        super_kwargs = {
            "exmsg": Scale._DEF_SCALE_EXC_MSG,
            "appmsg": Scale._DEF_SCALE_APP_MSG,
            "desc": Scale._DESC_PREFIX + desc_suffix,
        }
        self._kwargs = kwargs
        valid_super_kwargs = super()._init_kwargs()
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _fit_transform(self, df, verbose):
        cols_to_exclude = self._exclude_columns.copy()
        if self._exclude_obj_cols:
            obj_cols = list((df.dtypes[df.dtypes == object]).index)
            obj_cols = [x for x in obj_cols if x not in cols_to_exclude]
            cols_to_exclude += obj_cols
        self._col_order = list(df.columns)
        if cols_to_exclude:
            excluded = df[cols_to_exclude]
            apply_to = df[
                [col for col in df.columns if col not in cols_to_exclude]
            ]
        else:
            apply_to = df
        self._scaler = scaler_by_params(self.scaler, **self._kwargs)
        try:
            res = pd.DataFrame(
                data=self._scaler.fit_transform(apply_to),
                index=apply_to.index,
                columns=apply_to.columns,
            )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns {}".format(
                    apply_to.columns
                )
            )
        if cols_to_exclude:
            res = pd.concat([res, excluded], axis=1)
            res = res[self._col_order]
        self.is_fitted = True
        return res

    def _transform(self, df, verbose):
        cols_to_exclude = self._exclude_columns.copy()
        if self._exclude_obj_cols:
            obj_cols = list((df.dtypes[df.dtypes == object]).index)
            obj_cols = [x for x in obj_cols if x not in cols_to_exclude]
            cols_to_exclude += obj_cols
        self._col_order = list(df.columns)
        if cols_to_exclude:
            excluded = df[cols_to_exclude]
            apply_to = df[
                [col for col in df.columns if col not in cols_to_exclude]
            ]
        else:
            apply_to = df
        try:
            res = pd.DataFrame(
                data=self._scaler.transform(apply_to),
                index=apply_to.index,
                columns=apply_to.columns,
            )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns {}".format(
                    apply_to.columns
                )
            )
        if cols_to_exclude:
            res = pd.concat([res, excluded], axis=1)
            res = res[self._col_order]
        return res


class TfidfVectorizeTokenLists(PdPipelineStage):
    """A pipeline stage TFIDF-vectorizing a token-list column to count columns.

    Every cell in the input columns is assumed to be a list of strings, each
    representing a single token. The resulting TF-IDF vector is exploded into
    individual columns, each with the label 'lbl_i' where lbl is the original
    column label and i is the index of column in the count vector.

    The resulting columns are concatenated to the end of the dataframe.

    All valid sklearn.TfidfVectorizer keyword arguemnts can be provided as
    keyword arguments to the constructor, except 'input' and 'analyzer', which
    will be ignored. As usual, all valid PdPipelineStage constructor parameters
    can also be provided as keyword arguments.

    Parameters
    ----------
    column : str
        The label of the token-list column to TfIdf-vectorize.
    drop : bool, default True
        If set to True, the source column is dropped after being transformed.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[2, ['hovercraft', 'eels']], [5, ['eels', 'urethra']]]
        >>> df = pd.DataFrame(data, [1, 2], ['Age', 'tokens'])
        >>> tfvectorizer = pdp.TfidfVectorizeTokenLists('tokens')
        >>> tfvectorizer(df)
           Age  tokens_0  tokens_1  tokens_2
        1    2  0.579739  0.814802  0.000000
        2    5  0.579739  0.000000  0.814802
    """

    _DEF_CNTVEC_MSG = "Count-vectorizing column {}."

    def __init__(self, column, drop=True, **kwargs):
        self._column = column
        self._drop = drop
        msg = TfidfVectorizeTokenLists._DEF_CNTVEC_MSG.format(column)
        super_kwargs = {
            "exmsg": ("TfIdfVectorizeTokenLists precondition not met:"
                      "{} column not found.".format(column)),
            "appmsg": "{}..".format(msg),
            "desc": msg,
        }
        valid_vectorizer_args = _get_args_list(TfidfVectorizer.__init__)
        self._vectorizer_args = {
            k: kwargs[k] for k in kwargs
            if k in valid_vectorizer_args and k not in [
                'input', 'analyzer', 'self',
            ]
        }
        pipeline_stage_args = {
            k: kwargs[k] for k in kwargs
            if k not in valid_vectorizer_args
        }
        super_kwargs.update(**pipeline_stage_args)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._column in df.columns

    def _fit_transform(self, df, verbose):
        self._tfidf_vectorizer = TfidfVectorizer(
            input='content',
            analyzer=lambda x: x,
            **self._vectorizer_args,
        )
        vectorized = self._tfidf_vectorizer.fit_transform(df[self._column])
        self._n_features = vectorized.shape[1]
        self._res_col_names = [
            '{}_{}'.format(self._column, i)
            for i in range(self._n_features)
        ]
        vec_df = pd.DataFrame.sparse.from_spmatrix(
            data=vectorized, index=df.index, columns=self._res_col_names)
        inter_df = pd.concat([df, vec_df], axis=1)
        self.is_fitted = True
        if self._drop:
            return inter_df.drop(self._column, axis=1)
        return inter_df

    def _transform(self, df, verbose):
        vectorized = self._tfidf_vectorizer.transform(df[self._column])
        vec_df = pd.DataFrame.sparse.from_spmatrix(
            data=vectorized, columns=self._res_col_names)
        inter_df = pd.concat([df, vec_df], axis=1)
        if self._drop:
            return inter_df.drop(self._column, axis=1)
        return inter_df
