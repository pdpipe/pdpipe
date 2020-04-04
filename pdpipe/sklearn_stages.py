"""PdPipeline stages dependent on the scikit-learn Python library.

Please note that the scikit-learn Python package must be installed for the
stages in this module to work.

When attempting to load stages from this module, pdpipe will first attempt to
import sklearn. If it fails, it will issue a warning, will not import any of
the pipeline stages that make up this module, and continue to load other
pipeline stages.
"""

import inspect

import numpy as np
import pandas as pd
import sklearn.preprocessing
import tqdm
from skutil.preprocessing import scaler_by_params
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from pdpipe.core import PdPipelineStage, ColumnsBasedPipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.cq import OfDtypes
from pdpipe.shared import (
    _get_args_list,
    _identity_function,
)

from .exceptions import PipelineApplicationError


class Encode(ColumnsBasedPipelineStage):
    """A pipeline stage that encodes categorical columns to integer values.

    The encoder for each column is saved in the attribute `encoders`, which
    is a dict mapping each encoded column name to the
    sklearn.preprocessing.LabelEncoder object used to encode it.

    Parameters
    ----------
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be converted, except
        those given in the exclude_columns parameter. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See pdpipe.cq.
    exclude_columns : str or list-like, default None
        Label or labels of columns to be excluded from encoding. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
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

    def _transformation(self, df, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, df, verbose):
        self.encoders = {}
        columns_to_encode = self._get_columns(df, fit=True)
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


class Scale(ColumnsBasedPipelineStage):
    """A pipeline stage that scales data.

    Parameters
    ----------
    scaler : str
        The type of scaler to use to scale the data. One of 'StandardScaler',
        'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer'
        and 'Normalizer'.
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be scale. If columns is None then
        all columns of numeric dtype will be scaled, except those given in the
        exclude_columns parameter. Alternatively, this parameter can be
        assigned a callable returning an iterable of labels from an input
        pandas.DataFrame. See pdpipe.cq.
    exclude_columns : str or list-like, optional
        Label or labels of columns to be excluded from encoding. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See pdpipe.cq.
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

    def __init__(
        self,
        scaler,
        columns=None,
        exclude_columns=None,
        **kwargs
    ):
        self.scaler = scaler
        self._kwargs = kwargs
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
            'desc_temp': "Scale columns {}",
        }
        valid_super_kwargs = super()._init_kwargs()
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
        super_kwargs['none_columns'] = OfDtypes([np.number])
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        raise NotImplementedError

    def _fit_transform(self, df, verbose):
        self._columns_to_scale = self._get_columns(df, fit=True)
        unscaled_cols = [
            x for x in df.columns
            if x not in self._columns_to_scale
        ]
        col_order = list(df.columns)
        inter_df = df[self._columns_to_scale]
        self._scaler = scaler_by_params(self.scaler, **self._kwargs)
        try:
            inter_df = pd.DataFrame(
                data=self._scaler.fit_transform(inter_df.values),
                index=inter_df.index,
                columns=inter_df.columns,
            )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns {}".format(
                    self._columns_to_scale
                )
            )
        if len(unscaled_cols) > 0:
            unscaled = df[unscaled_cols]
            inter_df = pd.concat([inter_df, unscaled], axis=1)
            inter_df = inter_df[col_order]
        self.is_fitted = True
        return inter_df

    def _transform(self, df, verbose):
        unscaled_cols = [
            x for x in df.columns
            if x not in self._columns_to_scale
        ]
        col_order = list(df.columns)
        inter_df = df[self._columns_to_scale]
        try:
            inter_df = pd.DataFrame(
                data=self._scaler.transform(inter_df.values),
                index=inter_df.index,
                columns=inter_df.columns,
            )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns {}".format(
                    self._columns_to_scale
                )
            )
        if len(unscaled_cols) > 0:
            unscaled = df[unscaled_cols]
            inter_df = pd.concat([inter_df, unscaled], axis=1)
            inter_df = inter_df[col_order]
        return inter_df


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
    hierarchical_labels : bool, default False
        If set to True, the labels of resulting columns are of the form 'P_F'
        where P is the label of the original token-list column and F is the
        feature name (i.e. the string token it corresponds to). Otherwise, it
        is simply the feature name itself. If you plan to have two different
        TfidfVectorizeTokenLists pipeline stages vectorizing two different
        token-list columns, you should set this to true, so tf-idf features
        originating in different text columns do not overwrite one another.

    Example
    -------
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
            "exmsg": ("TfIdfVectorizeTokenLists precondition not met:"
                      "{} column not found.".format(column)),
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
            analyzer=_identity_function,
            **self._vectorizer_args,
        )
        vectorized = self._tfidf_vectorizer.fit_transform(df[self._column])
        self._n_features = vectorized.shape[1]
        if self._hierarchical_labels:
            self._res_col_names = [
                '{}_{}'.format(self._column, f)
                for f in self._tfidf_vectorizer.get_feature_names()
            ]
        else:
            self._res_col_names = self._tfidf_vectorizer.get_feature_names()
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
            data=vectorized, index=df.index, columns=self._res_col_names)
        inter_df = pd.concat([df, vec_df], axis=1)
        if self._drop:
            return inter_df.drop(self._column, axis=1)
        return inter_df


class _BaseSklearnTransformer(ColumnsBasedPipelineStage):
    """Apply a scikit-learn Transformer to a set of columns.

    The transformer object is saved in the attribute `transformer`.

    The transformed columns are concatenared to the input dataframe, unless
    `drop` is set to `True` AND `res_label_format` is set to `suffix`, in which
    case they replace the source columns that were transformed.

    Parameters
    ----------
    transformer : sklearn.TransformerMixin object or class
        A class or instance of an sklearn.TransformerMixin.
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be transformed. If columns is None
        then all columns are transformed, except those given in the
        exclude_columns parameter. Alternatively, this parameter can be
        assigned a callable returning an iterable of labels from an input
        pandas.DataFrame. See pdpipe.cq.
    exclude_columns : str or list-like, default None
        Label or labels of columns to be excluded from transform. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
    drop : bool, default True
        If set to True, the source columns are dropped after being transformed.
        Otherwise, they are kept in their original position.
    res_label : str, default 'transformed'
        A string used in construction of resulting column labels.
    res_label_format : 'consecutive' or 'suffix', default 'consecutive'
        The format used to generate the labels of the resulting columns. The
        default, 'consecutive', simply uses the value of `res_label` with
        increasing numbers, as in 'foo_0', 'foo_1' and so on. Using 'suffix'
        instead assumes that the number of resulting columns is identical to
        the number of source columns  - and indeed that each resulting column
        is a transformation of the corresponding source column - and thus its
        behaviour depends on the `drop` parameter: If it is set to False, then
        `res_label` is used as a suffix, and is appended to the labels of the
        source columns to generate resulting column labels. If `drop` is set to
        `True` then the labels of the source columns are used verbatim.
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the PdPipelineStage
        constructor. All other keywords are forwarded to the constructor of the
        sklearm transformer class used.

    Example
    -------
    """

    def __init__(
        self,
        transformer,
        columns=None,
        exclude_columns=None,
        drop=True,
        res_label=None,
        res_label_format=None,
        **kwargs
    ):
        self._transformer_class = transformer
        if not inspect.isclass(transformer):
            self._transformer_class = type(transformer)
        self._drop = drop
        if res_label is None:
            res_label = 'transformed'
        self._res_label = res_label
        if res_label_format is None:
            res_label_format = 'consecutive'
        self._res_label_format = res_label_format
        self._kwargs = kwargs
        super_kwargs = {
            'columns': columns,
            'exclude_columns': exclude_columns,
        }
        valid_super_kwargs = super()._init_kwargs()
        self._transformer_kwargs = {}
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
            else:
                self._transformer_kwargs[key] = kwargs[key]
        super_kwargs['none_columns'] = 'all'
        super_kwargs['desc_temp'] = (
            "Transform columns {{}} with {} and kwargs {}".format(
                self._transformer_class, self._transformer_kwargs))
        super().__init__(**super_kwargs)

    def _transformation(self, df, verbose, fit):
        raise NotImplementedError

    def _get_column_labels(self, data, original_labels):
        num_cols = data.shape[1]
        if self._res_label_format == 'suffix':
            if num_cols != len(original_labels):
                raise PipelineApplicationError((
                    "scikit-learn transformater produced a different number of"
                    " columns ({}) than the number of input columns ({}) when "
                    "applied, while expected to produce the same number!"
                ).format(num_cols, len(original_labels)))
            if self._drop:
                return original_labels
            return [x + self._res_label for x in original_labels]
        return ['{}_{}'.format(self._res_label, i) for i in range(num_cols)]

    def _fit_transform(self, df, verbose):
        if verbose:
            print("Constructing transformer {} with kwargs {}...".format(
                self._transformer_class, self._transformer_kwargs))
        self._transformer_obj = self._transformer_class.__init__(
            **self._transformer_kwargs)
        self._columns_to_transform = self._get_columns(df, fit=True)
        if verbose:
            print("Transforming columns {}...".format(
                self._columns_to_transform))
        untransformed_cols = [
            x for x in df.columns
            if x not in self._columns_to_transform
        ]
        col_order = list(df.columns)
        inter_df = df[self._columns_to_transform]
        try:
            data = self._transformer_obj.fit_transform(inter_df.values)
        except Exception as e:
            raise PipelineApplicationError(
                (
                    "Exception raised when transformer {} applied to "
                    "columns {}").format(
                        self._transformer_obj,
                        self._columns_to_scale,
                ),
                e,
            )
        self._res_column_labels = self._get_column_labels(
            data, inter_df.columns)
        inter_df = pd.DataFrame(
            data=data,
            index=inter_df.index,
            columns=self._res_column_labels,
        )
        if self._drop:
            if len(untransformed_cols) > 0:
                untransformed = df[untransformed_cols]
                inter_df = pd.concat([untransformed, inter_df], axis=1)
                if self._res_label_format == 'suffix':
                    inter_df = inter_df[col_order]
        else:
            inter_df = pd.concat([df, inter_df], axis=1)
        self.is_fitted = True
        return inter_df

    def _transform(self, df, verbose):
        if verbose:
            print("Transforming columns {}...".format(
                self._columns_to_transform))
        untransformed_cols = [
            x for x in df.columns
            if x not in self._columns_to_transform
        ]
        col_order = list(df.columns)
        try:
            inter_df = df[self._columns_to_transform]
        except KeyError:
            raise PipelineApplicationError((
                "Columns missing from input dataframe on transform! "
                "Expected columns: {}".format(self._columns_to_transform)))
        try:
            data = self._transformer_obj.transform(inter_df.values)
        except Exception as e:
            raise PipelineApplicationError(
                (
                    "Exception raised when transformer {} applied to "
                    "columns {}").format(
                        self._transformer_obj,
                        self._columns_to_scale,
                ),
                e,
            )
        inter_df = pd.DataFrame(
            data=data,
            index=inter_df.index,
            columns=self._res_column_labels,
        )
        if self._drop:
            if len(untransformed_cols) > 0:
                untransformed = df[untransformed_cols]
                inter_df = pd.concat([untransformed, inter_df], axis=1)
                if self._res_label_format == 'suffix':
                    inter_df = inter_df[col_order]
        else:
            inter_df = pd.concat([df, inter_df], axis=1)
        return inter_df


class SklearnTransformer(_BaseSklearnTransformer):
    """Apply a scikit-learn Transformer to a set of columns.

    The transformer object is saved in the attribute `transformer`.

    The transformed columns are concatenared to the input dataframe.

    Parameters
    ----------
    transformer : sklearn.TransformerMixin object or class
        A class or instance of an sklearn.TransformerMixin.
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be transformed. If columns is None
        then all columns are transformed, except those given in the
        exclude_columns parameter. Alternatively, this parameter can be
        assigned a callable returning an iterable of labels from an input
        pandas.DataFrame. See pdpipe.cq.
    exclude_columns : str or list-like, default None
        Label or labels of columns to be excluded from transform. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
    drop : bool, default True
        If set to True, the source columns are dropped after being transformed.
        Otherwise, they are kept in their original position.
    res_label : str, default None
        The prefix to use for labels of resulting columns. If not given, the
        transformer name is used as default. For example, providing 'foo' will
        result in columns 'foo_0', 'foo_1', etc.
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the transfomer
        constructor on scaler creation (e.g. 'n_quantiles' for
        QuantileTransformer). All other extra keyword arguments are assumed to
        be valid PdPipelineStage constructor arguments.

    Example
    -------
    """

    def __init__(
        self,
        transformer,
        columns=None,
        exclude_columns=None,
        drop=True,
        res_label=None,
        **kwargs
    ):
        if not inspect.isclass(transformer):
            transformer = type(transformer)
        if res_label is None:
            res_label = repr(transformer)
        super().__init__(
            transformer=transformer,
            columns=columns,
            exclude_columns=exclude_columns,
            drop=drop,
            res_label=res_label,
            res_label_format='consecutive',
            **kwargs,
        )


class InjectiveSklearnTransformer(_BaseSklearnTransformer):
    """Apply a scikit-learn Transformer to a set of columns.

    The transformer object is saved in the attribute `transformer`.

    The transformed columns are concatenated to the input dataframe.

    Parameters
    ----------
    transformer : sklearn.TransformerMixin object or class
        A class or instance of an sklearn.TransformerMixin.
    columns : single label, list-like or callable, default None
        Column labels in the DataFrame to be transformed. If columns is None
        then all columns are transformed, except those given in the
        exclude_columns parameter. Alternatively, this parameter can be
        assigned a callable returning an iterable of labels from an input
        pandas.DataFrame. See pdpipe.cq.
    exclude_columns : str or list-like, default None
        Label or labels of columns to be excluded from transform. If None then
        no column is excluded. Alternatively, this parameter can be assigned a
        callable returning an iterable of labels from an input
    drop : bool, default True
        If set to True, the source columns are dropped after being transformed,
        and the corresponding transformed columns take their place. Otherwise,
        they are kept in their original position, and the resulting columns are
        concatenated to the end of the input dataframe.
    res_label : str, default None
        The suffix added to the labels of source columns to generate the labels
        of the to resulting columns. If not given, the string 'transformed' is
        used as default. For example, providing 'foo' will result in columns
        'a_foo', 'b_foo', etc for source columnes 'a', 'b' and so on.
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the transfomer
        constructor on creation (e.g. 'n_quantiles' for QuantileTransformer).
        All other extra keyword arguments are assumed to be valid
        PdPipelineStage constructor arguments.

    Example
    -------
    """

    def __init__(
        self,
        transformer,
        columns=None,
        exclude_columns=None,
        drop=True,
        res_label=None,
        **kwargs
    ):
        super().__init__(
            transformer=transformer,
            columns=columns,
            exclude_columns=exclude_columns,
            drop=drop,
            res_label=res_label,
            res_label_format='suffix',
            **kwargs,
        )
