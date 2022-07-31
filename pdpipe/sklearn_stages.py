"""PdPipeline stages dependent on the scikit-learn Python library.

Please note that the scikit-learn Python package must be installed for the
stages in this module to work.

When attempting to load stages from this module, pdpipe will first attempt to
import sklearn. If it fails, it will issue a warning, will not import any of
the pipeline stages that make up this module, and continue to load other
pipeline stages.
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing
from tqdm.autonotebook import tqdm
from skutil.preprocessing import scaler_by_params
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.base import clone

from pdpipe.core import PdPipelineStage, ColumnsBasedPipelineStage
from pdpipe.util import (
    out_of_place_col_insert,
    per_column_values_sklearn_transform,
)
from pdpipe.cq import OfDtypes
from pdpipe.shared import (
    _get_args_list,
    _identity_function,
)

from .exceptions import (
    PipelineApplicationError,
    UnfittedPipelineStageError,
    UnexpectedPipelineMethodCallError,
)
from .lbl import _SkipOnLabelPlaceholderPredict


class Encode(ColumnsBasedPipelineStage):
    """A pipeline stage that encodes categorical columns to integer values.

    The encoder for each column is saved in the attribute 'encoders', which
    is a dict mapping each encoded column name to the
    sklearn.preprocessing.LabelEncoder object used to encode it.

    Parameters
    ----------
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be converted, except
        those given in the exclude_columns parameter. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`.
    exclude_columns : single label, list-like or callable, default None
        Label or labels of columns to be excluded from encoding. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
        pandas.DataFrame. See `pdpipe.cq`.
    drop : bool, default True
        If set to True, the source columns are dropped after being encoded,
        and the resulting encoded columns retain the names of the source
        columns. Otherwise, encoded columns gain the suffix '_enc'.

    Attributes
    ----------
    encoders : dict
        A dictionary mapping each encoded column name to the corresponding
        sklearn.preprocessing.LabelEncoder object. Empty object if not fitted.

    Examples
    --------
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

    def __init__(
        self, columns=None, exclude_columns=None, drop=True, **kwargs
    ):
        self._drop = drop
        self.encoders = {}
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
            'desc_temp': "Encode {}",
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = OfDtypes(['object', 'category'])
        super().__init__(**super_kwargs)

    def _transformation(self, X, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, X, verbose):
        self.encoders = {}
        columns_to_encode = self._get_columns(X, fit=True)
        if verbose:
            columns_to_encode = tqdm(columns_to_encode)
        inter_X = X
        for colname in columns_to_encode:
            lbl_enc = sklearn.preprocessing.LabelEncoder()
            source_col = X[colname]
            loc = X.columns.get_loc(colname) + 1
            new_name = colname + "_enc"
            if self._drop:
                inter_X = inter_X.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_X = out_of_place_col_insert(
                X=inter_X,
                series=lbl_enc.fit_transform(source_col),
                loc=loc,
                column_name=new_name,
            )
            self.encoders[colname] = lbl_enc
        self.is_fitted = True
        return inter_X

    def _transform(self, X, verbose):
        inter_X = X
        for colname in self.encoders:
            lbl_enc = self.encoders[colname]
            source_col = X[colname]
            loc = X.columns.get_loc(colname) + 1
            new_name = colname + "_enc"
            if self._drop:
                inter_X = inter_X.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_X = out_of_place_col_insert(
                X=inter_X,
                series=lbl_enc.transform(source_col),
                loc=loc,
                column_name=new_name,
            )
        return inter_X


class Scale(ColumnsBasedPipelineStage):
    """A pipeline stage that scales data.

    Parameters
    ----------
    scaler : str
        The type of scaler to use to scale the data. One of 'StandardScaler',
        'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer'
        and 'Normalizer'. Refer to scikit-learn's documentation for usage.
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be scaled. If columns is None then
        all columns of numeric dtype will be scaled, except those given in the
        exclude_columns parameter. Alternatively, this parameter can be
        assigned a callable returning an iterable of labels from an input
        pandas.DataFrame. See `pdpipe.cq`.
    exclude_columns : single label, list-like or callable, default None
        Label or labels of columns to be excluded from encoding. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`.
    joint : bool, default False
        If set to True, all scaled columns will be scaled as a single value
        set (meaning, only the single largest value among all input columns
        will be scaled to 1, and not the largest one for each column).
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the scaler
        constructor on scaler creation (e.g. 'n_quantiles' for
        QuantileTransformer). PdPipelineStage valid keyword arguments are used
        to override Scale class defaults.

    Attributes
    ----------
    scaler : sklearn._OneToOneFeatureMixin
        A scikit-learn scaler object.

    Examples
    --------
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

    def __init__(
        self,
        scaler,
        columns=None,
        exclude_columns=None,
        joint=False,
        **kwargs
    ):
        self.scaler = scaler
        self._joint = joint
        self._kwargs = kwargs.copy()
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
            'desc_temp': "Scale columns {}",
        }
        valid_super_kwargs = super()._init_kwargs()
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
                self._kwargs.pop(key)
        super_kwargs['none_columns'] = OfDtypes([np.number])
        super().__init__(**super_kwargs)

    def _transformation(self, X, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, X, verbose):
        self._columns_to_scale = self._get_columns(X, fit=True)
        unscaled_cols = [
            x for x in X.columns
            if x not in self._columns_to_scale
        ]
        col_order = list(X.columns)
        inter_X = X[self._columns_to_scale]
        self._scaler = scaler_by_params(self.scaler, **self._kwargs)
        try:
            if self._joint:
                self._scaler.fit(np.array([inter_X.values.flatten()]).T)
                inter_X = per_column_values_sklearn_transform(
                    X=inter_X,
                    transform=self._scaler.transform
                )
            else:
                inter_X = pd.DataFrame(
                    data=self._scaler.fit_transform(inter_X.values),
                    index=inter_X.index,
                    columns=inter_X.columns,
                )
        except Exception as e:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns"
                f" {self._columns_to_scale} by class {self.__class__}"
            ) from e
        if len(unscaled_cols) > 0:
            unscaled = X[unscaled_cols]
            inter_X = pd.concat([inter_X, unscaled], axis=1)
            inter_X = inter_X[col_order]
        self.is_fitted = True
        return inter_X

    def _transform(self, X, verbose):
        unscaled_cols = [
            x for x in X.columns
            if x not in self._columns_to_scale
        ]
        col_order = list(X.columns)
        inter_X = X[self._columns_to_scale]
        try:
            if self._joint:
                inter_X = per_column_values_sklearn_transform(
                    X=inter_X,
                    transform=self._scaler.transform
                )
            else:
                inter_X = pd.DataFrame(
                    data=self._scaler.transform(inter_X.values),
                    index=inter_X.index,
                    columns=inter_X.columns,
                )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns"
                f" {self._columns_to_scale} by class {self.__class__}"
            )
        if len(unscaled_cols) > 0:
            unscaled = X[unscaled_cols]
            inter_X = pd.concat([inter_X, unscaled], axis=1)
            inter_X = inter_X[col_order]
        return inter_X


class TfidfVectorizeTokenLists(PdPipelineStage):
    """A pipeline stage TFIDF-vectorizing a token-list column to count columns.

    Every cell in the input columns is assumed to be a list of strings, each
    representing a single token. The resulting TF-IDF vector is exploded into
    individual columns, each with the label 'lbl_i' where lbl is the original
    column label and i is the index of column in the count vector.

    The resulting columns are concatenated to the end of the dataframe.

    All valid sklearn.feature_extraction.text.TfidfVectorizer keyword arguments
    can be provided as keyword arguments to the constructor, except 'input' and
    'analyzer', which will be ignored. As usual, all valid PdPipelineStage
    constructor parameters can also be provided as keyword arguments.

    Parameters
    ----------
    column : str
        The label of the token-list column to TfIdf-vectorize.
    drop : bool, default True
        If set to True, the source column is dropped after being transformed.
    hierarchical_labels : bool, default False
        If set to True, the labels of resulting columns are of the form 'P_F'
        where P is the label of the original token-list column and F is the
        feature name (i.e. the string token it corresponds to). Otherwise, it
        is simply the feature name itself. If you plan to have two different
        TfidfVectorizeTokenLists pipeline stages vectorizing two different
        token-list columns, you should set this to true, so tf-idf features
        originating in different text columns do not overwrite one another.

    Examples
    --------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[2, ['hovercraft', 'eels']], [5, ['eels', 'urethra']]]
    >>> df = pd.DataFrame(data, [1, 2], ['Age', 'tokens'])
    >>> tfvectorizer = pdp.TfidfVectorizeTokenLists('tokens')
    >>> tfvectorizer(df)
       Age      eels  hovercraft   urethra
    1    2  0.579739    0.814802  0.000000
    2    5  0.579739    0.000000  0.814802
    """

    _DEF_CNTVEC_MSG = "Count-vectorizing column {}."

    def __init__(self, column, drop=True, hierarchical_labels=False,
                 **kwargs):
        self._column = column
        self._drop = drop
        self._hierarchical_labels = hierarchical_labels
        msg = TfidfVectorizeTokenLists._DEF_CNTVEC_MSG.format(column)
        super_kwargs = {
            "exmsg": ("TfidfVectorizeTokenLists precondition not met:"
                      f"{column} column not found."),
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
            if k in PdPipelineStage._INIT_KWARGS
        }
        super_kwargs.update(**pipeline_stage_args)
        super().__init__(**super_kwargs)

    def _prec(self, X):
        return self._column in X.columns

    def _fit_transform(self, X, verbose):
        self._tfidf_vectorizer = TfidfVectorizer(
            input='content',
            analyzer=_identity_function,
            **self._vectorizer_args,
        )
        vectorized = self._tfidf_vectorizer.fit_transform(X[self._column])
        self._n_features = vectorized.shape[1]
        if self._hierarchical_labels:
            self._res_col_names = [
                f'{self._column}_{f}'
                for f in self._tfidf_vectorizer.get_feature_names_out()
            ]
        else:
            self._res_col_names = self._tfidf_vectorizer.get_feature_names_out(
            )
        vec_X = pd.DataFrame.sparse.from_spmatrix(
            data=vectorized, index=X.index, columns=self._res_col_names)
        inter_X = pd.concat([X, vec_X], axis=1)
        self.is_fitted = True
        if self._drop:
            return inter_X.drop(self._column, axis=1)
        return inter_X

    def _transform(self, X, verbose):
        vectorized = self._tfidf_vectorizer.transform(X[self._column])
        vec_X = pd.DataFrame.sparse.from_spmatrix(
            data=vectorized, index=X.index, columns=self._res_col_names)
        inter_X = pd.concat([X, vec_X], axis=1)
        if self._drop:
            return inter_X.drop(self._column, axis=1)
        return inter_X


class Decompose(ColumnsBasedPipelineStage):
    """A pipeline stage applying dimensionality reduction through matrix
    decomposition.

    Parameters
    ----------
    transformer : sklearn.TransformerMixin
        An instance of a matrix decomposer transformer from the
        `sklearn. decomposition` module.
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be transformer. If columns is None
        then all columns of numeric dtype will be scaled, except those given
        in the exclude_columns parameter. Alternatively, this parameter can be
        assigned a callable returning an iterable of labels from an input
        pandas.DataFrame. See `pdpipe.cq`.
    exclude_columns : single label, list-like or callable, default None
        Label or labels of columns to be excluded from encoding. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`.
    drop : bool, default True
        If set to True, decomposed columns are dropped, and the resulting set
        of columns are concatenated to all un-transformed columns, with
        matching column labels (see the `lbl_format` parameters). If set to
        False, the new columns are instead concatenated to the input dataframe.
    lbl_format : str, optional
        An f-string with a single {} slot, used to generated post-decomposition
        column labels. For example, 'pca{:0>3}' will yield columns 'pca000',
        'pca001', etc. If not provided, the default 'mdc{}' is used.
    **kwargs : extra keyword arguments
        PdPipelineStage valid keyword arguments are used to override
        Decompose class defaults. All other extra keyword arguments are
        forwarded to the transformer constructor on transformer creation.

    Attributes
    ----------
    transformer : sklearn.TransformerMixin
        The transformer instance used to perform the decomposition.

    Examples
    --------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> from sklearn.decomposition import PCA
    >>> data = [[3, 1, 1], [7, 2, 4], [8, 3, 1]]
    >>> df = pd.DataFrame(data, [1,2,3], ["a", "b", "c"])
    >>> pca_stage = pdp.Decompose(PCA(), n_components=2)
    >>> pca_stage(df)
           mdc0      mdc1
    1  3.313301 -0.148453
    2 -1.432127  1.717269
    3 -1.881174 -1.568816
    """

    def __init__(
        self,
        transformer,
        columns=None,
        exclude_columns=None,
        drop=True,
        lbl_format=None,
        **kwargs
    ):
        self.transformer = transformer
        self._drop = drop
        self._lbl_format = lbl_format
        if lbl_format is None:
            self._lbl_format = 'mdc{}'
        self._kwargs = kwargs.copy()
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
            'desc_temp': f"Decompose columns {{}} with {transformer}",
        }
        valid_super_kwargs = super()._init_kwargs()
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
                self._kwargs.pop(key)
        super_kwargs['none_columns'] = OfDtypes([np.number])
        super().__init__(**super_kwargs)

    def _transformation(self, X, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, X, verbose):
        self._columns_to_transform = self._get_columns(X, fit=True)
        untransformed_cols = [
            x for x in X.columns
            if x not in self._columns_to_transform
        ]
        sub_X = X[self._columns_to_transform]
        self._transformer = clone(self.transformer)
        self._transformer = self._transformer.set_params(**self._kwargs)
        try:
            inter_X = self._transformer.fit_transform(sub_X.values)
            n_cols = inter_X.shape[1]
            columns = [
                self._lbl_format.format(i)
                for i in range(n_cols)
            ]
            inter_X = pd.DataFrame(
                data=inter_X,
                index=X.index,
                columns=columns,
            )
        except Exception as e:
            raise PipelineApplicationError(
                "Exception raised when Decompose applied to columns"
                f" {self._columns_to_transform} by class {self.__class__}"
            ) from e
        if self._drop:
            if len(untransformed_cols) > 0:
                untransformed = X[untransformed_cols]
                inter_X = pd.concat([untransformed, inter_X], axis=1)
        else:
            inter_X = pd.concat([X, inter_X], axis=1)
        self.is_fitted = True
        return inter_X

    def _transform(self, X, verbose):
        untransformed_cols = [
            x for x in X.columns
            if x not in self._columns_to_transform
        ]
        sub_X = X[self._columns_to_transform]
        try:
            inter_X = self._transformer.transform(sub_X.values)
            n_cols = inter_X.shape[1]
            columns = [
                self._lbl_format.format(i)
                for i in range(n_cols)
            ]
            inter_X = pd.DataFrame(
                data=inter_X,
                index=X.index,
                columns=columns,
            )
        except Exception as e:
            raise PipelineApplicationError(
                "Exception raised when Decompose applied to columns"
                f" {self._columns_to_transform} by class {self.__class__}"
            ) from e
        if self._drop:
            if len(untransformed_cols) > 0:
                untransformed = X[untransformed_cols]
                inter_X = pd.concat([untransformed, inter_X], axis=1)
        else:
            inter_X = pd.concat([X, inter_X], axis=1)
        return inter_X


class EncodeLabel(PdPipelineStage):
    """A pipeline stage that encodes the input label series to integer values.

    The encoder for each column is saved in the attribute 'encoder', which
    is a dict mapping each encoded column name to the
    The used `sklearn.preprocessing.LabelEncoder` object is saved in the
    `encoder_` attribute.

    Attributes
    ----------
    encoder_ : sklearn.preprocessing.LabelEncoder
        The sklearn.preprocessing.LabelEncoder object used to encode the series
        label.

    Examples
    --------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, 31], [7.2, 33], [12.1, 28]]
    >>> X = pd.DataFrame(data, [1,2,3], ["ph","temp"])
    >>> y = pd.Series(["acd", "alk", "alk"])
    >>> encode_stage = pdp.EncodeLabel()
    >>> X, y = encode_stage(X, y)
    >>> X
         ph  temp
    1   3.2    31
    2   7.2    33
    3  12.1    28
    >>> y
    1    0
    2    1
    3    1
    dtype: int...
    >>> encode_stage.encoder_.inverse_transform([0,1,1])
    array(['acd', 'alk', 'alk'], dtype=object)
    """

    def __init__(self, **kwargs: object) -> None:
        skipi = _SkipOnLabelPlaceholderPredict()
        if 'skip' in kwargs:
            skipi.skip_cond = kwargs.pop('skip')
        super_kwargs = {
            'desc': "Encode label values",
            'skip': skipi,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, X, y):
        return y is not None

    def _transform(self, X, verbose):
        raise UnexpectedPipelineMethodCallError(  # pragma: no cover
            "EncodeLabel._transform() is not expected to be called!")

    def _fit_transform_Xy(self, X, y, verbose):
        self.encoder_ = sklearn.preprocessing.LabelEncoder()
        post_y = self.encoder_.fit_transform(y)
        post_y = pd.Series(data=post_y, index=y.index)
        self.is_fitted = True
        return X, post_y

    def _transform_Xy(self, X, y, verbose):
        try:
            post_y = self.encoder_.transform(y)
            post_y = pd.Series(data=post_y, index=y.index)
            return X, post_y
        except AttributeError:
            raise UnfittedPipelineStageError("EncodeLabel is not fitted!")
