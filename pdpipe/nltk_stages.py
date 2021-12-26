"""PdPipeline stages dependent on the nltk Python library.

Please note that the nltk Python package must be installed for the stages in
this module to work.

When attempting to load stages from this module, pdpipe will first attempt to
import nltk. If it fails, it will issue a warning, will not import any of the
pipeline stages that make up this module, and continue to load other pipeline
stages.
"""

import os
import importlib
try:
    from collections.abc import Iterable
except ImportError:  # pragma: no cover:
    from collections import Iterable

import nltk
import pandas as pd
import tqdm

from pdpipe.core import ColumnsBasedPipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.col_generation import MapColVals
from pdpipe.shared import (
    _interpret_columns_param,
    _list_str
)


class TokenizeText(MapColVals):
    """A pipeline stage that tokenizes a text column into token lists.

    Note: The nltk package must be installed for this pipeline stage to work.

    Parameters
    ----------
    columns : single label, list-like of callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    drop : bool, default True
        If set to True, the source columns are dropped after being tokenized,
        and the resulting tokenized columns retain the names of the source
        columns. Otherwise, tokenized columns gain the suffix '_tok'.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame(
        ...     [[3.2, "Kick the baby!"]], [1], ['freq', 'content'])
        >>> tokenize_stage = pdp.TokenizeText('content')
        >>> tokenize_stage(df)
           freq               content
        1   3.2  [Kick, the, baby, !]
    """

    _DEF_TOKENIZE_EXC_MSG = ("Tokenize stage failed because not all columns "
                             "{} are present in input dataframe and are of"
                             " dtype object.")
    _DEF_TOKENIZE_APP_MSG = "Tokenizing {}..."

    @staticmethod
    def __check_punkt():
        try:
            nltk.word_tokenize('a a')
        except LookupError:  # pragma: no cover
            # try:
            #     nltk.data.find('corpora/stopwords')
            # except LookupError:  # pragma: no cover
            dpath = os.path.expanduser('~/nltk_data/tokenizers')
            os.makedirs(dpath, exist_ok=True)
            nltk.download('punkt')

    def __init__(self, columns, drop=True, **kwargs):
        self.__check_punkt()
        self._columns = _interpret_columns_param(columns)
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': nltk.word_tokenize,
            'drop': drop,
            'suffix': '_tok',
            'exmsg': TokenizeText._DEF_TOKENIZE_EXC_MSG.format(col_str),
            'desc': f"Tokenize {col_str}",
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'error'
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class UntokenizeText(MapColVals):
    """A pipeline stage that joins token lists to whitespace-separated strings.

    Target columns must be series of token lists; i.e. every cell in the series
    is an iterable of string tokens.

    Note: The nltk package must be installed for this pipeline stage to work.

    Parameters
    ----------
    columns : single label, list-like of callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    drop : bool, default True
        If set to True, the source columns are dropped after being untokenized,
        and the resulting columns retain the names of the source columns.
        Otherwise, untokenized columns gain the suffix '_untok'.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3.2, ['Shake', 'and', 'bake!']]]
        >>> df = pd.DataFrame(data, [1], ['freq', 'content'])
        >>> untokenize_stage = pdp.UntokenizeText('content')
        >>> untokenize_stage(df)
           freq          content
        1   3.2  Shake and bake!
    """

    _DEF_UNTOKENIZE_EXC_MSG = (
        "Untokenize stage failed because not all columns {} are present in "
        "input dataframe and are of dtype object.")

    @staticmethod
    def _untokenize_list(token_list):
        return ' '.join(token_list)

    def __init__(self, columns, drop=True, **kwargs):
        self._columns = _interpret_columns_param(columns)
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': UntokenizeText._untokenize_list,
            'drop': drop,
            'suffix': '_untok',
            'exmsg': UntokenizeText._DEF_UNTOKENIZE_EXC_MSG.format(col_str),
            'desc': f"Untokenize {col_str}",
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'error'
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class RemoveStopwords(MapColVals):
    """A pipeline stage that removes stopwords from a tokenized list.

    Target columns must be series of token lists; i.e. every cell in the series
    is an iterable of string tokens.

    Note: The nltk package must be installed for this pipeline stage to work.

    Parameters
    ----------
    language : str or array-like
        If a string is given, interpreted as the language of the stopwords, and
        should then be one of the languages supported by the NLTK Stopwords
        Corpus. If a list is given, it is assumed to be the list of stopwords
        to remove.
    columns : single label, list-like or callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    drop : bool, default True
        If set to True, the source columns are dropped after stopword removal,
        and the resulting columns retain the names of the source columns.
        Otherwise, resulting columns gain the suffix '_nostop'.

    Example
    -------
        >> import pandas as pd; import pdpipe as pdp;
        >> data = [[3.2, ['kick', 'the', 'baby']]]
        >> df = pd.DataFrame(data, [1], ['freq', 'content'])
        >> remove_stopwords = pdp.RemoveStopwords('english', 'content')
        >> remove_stopwords(df)
           freq       content
        1   3.2  [kick, baby]
    """

    _DEF_STOPWORDS_EXC_MSG = ("RemoveStopwords stage failed because not all "
                              "columns {} are present in input dataframe and "
                              "are of dtype object.")
    _DEF_STOPWORDS_APP_MSG = "Removing stopwords from {}..."

    class _StopwordsRemover(object):
        def __init__(self, stopwords_list):
            self.stopwords_list = stopwords_list

        def __call__(self, word_list):
            return [w for w in word_list if w not in self.stopwords_list]

    @staticmethod
    def __stopwords_by_language(language):
        try:
            from nltk.corpus import stopwords
            return stopwords.words(language)
        except LookupError:  # pragma: no cover
            # try:
            #     nltk.data.find('corpora/stopwords')
            # except LookupError:  # pragma: no cover
            dpath = os.path.expanduser('~/nltk_data/corpora/stopwords')
            os.makedirs(dpath, exist_ok=True)
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            return stopwords.words(language)

    def __init__(self, language, columns, drop=True, **kwargs):
        self._language = language
        if isinstance(language, str):
            self._stopwords_list = RemoveStopwords.__stopwords_by_language(
                language)
        elif isinstance(language, Iterable):
            self._stopwords_list = list(language)
        else:
            raise TypeError("language parameter should be string or list!")
        self._stopwords_remover = RemoveStopwords._StopwordsRemover(
            self._stopwords_list)
        self._columns = _interpret_columns_param(columns)
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': self._stopwords_remover,
            'drop': drop,
            'suffix': '_nostop',
            'exmsg': RemoveStopwords._DEF_STOPWORDS_EXC_MSG.format(col_str),
            'desc': f"Remove stopwords from {col_str}",
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'error'
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class SnowballStem(MapColVals):
    """A pipeline stage that stems tokens in a list using the Snowball stemmer.

    Target columns must be series of token lists; i.e. every cell in the series
    is an iterable of string tokens.

    Note: The nltk package must be installed for this pipeline stage to work.

    Parameters
    ----------
    stemmer_name : str
        The name of the Snowball stemmer to use. Should be one of the Snowball
        stemmers implemented by nltk. E.g. 'EnglishStemmer'.
    columns : single label, list-like or callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    drop : bool, default True
        If set to True, the source columns are dropped after stemming, and the
        resulting columns retain the names of the source columns. Otherwise,
        resulting columns gain the suffix '_stem'.
    min_len : int, optional
        If provided, tokens shorter than this length are not stemmed.
    max_len : int, optional
        If provided, tokens longer than this length are not stemmed.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3.2, ['kicking', 'boats']]]
        >>> df = pd.DataFrame(data, [1], ['freq', 'content'])
        >>> remove_stopwords = pdp.SnowballStem('EnglishStemmer', 'content')
        >>> remove_stopwords(df)
           freq       content
        1   3.2  [kick, boat]
    """

    _DEF_STEM_EXC_MSG = ("SnowballStem stage failed because not all "
                         "columns {} are present in input dataframe and "
                         "are of dtype object.")
    _DEF_STEM_DESC = "Stemming tokens{} in {}..."

    class MinLenStemCondition(object):

        def __init__(self, min_len):
            self.min_len = min_len

        def __call__(self, x):
            return len(x) >= self.min_len

    class MaxLenStemCondition(object):

        def __init__(self, max_len):
            self.max_len = max_len

        def __call__(self, x):
            return len(x) <= self.max_len

    class MinMaxLenStemCondition(object):

        def __init__(self, min_len, max_len):
            self.min_len = min_len
            self.max_len = max_len

        def __call__(self, x):
            return (len(x) >= self.min_len) and (len(x) <= self.max_len)

    class _TokenListStemmer(object):
        def __init__(self, stemmer, min_len=None, max_len=None):
            self.stemmer = stemmer
            self.cond = None
            if min_len:
                if max_len:
                    self.cond = SnowballStem.MinMaxLenStemCondition(
                        min_len=min_len, max_len=max_len)
                else:
                    self.cond = SnowballStem.MinLenStemCondition(min_len)
            elif max_len:
                self.cond = SnowballStem.MaxLenStemCondition(max_len)
            self.__stem__ = self.__uncond_stem__
            if self.cond:
                self.__stem__ = self.__cond_stem__

        def __call__(self, token_list):
            return self.__stem__(token_list)

        def __uncond_stem__(self, token_list):
            return [self.stemmer.stem(w) for w in token_list]

        def __cond_stem__(self, token_list):
            return [
                self.stemmer.stem(w) if self.cond(w) else w
                for w in token_list
            ]

    @staticmethod
    def __stemmer_by_name(stemmer_name):
        snowball_module = importlib.import_module('nltk.stem.snowball')
        stemmer_cls = getattr(snowball_module, stemmer_name)
        return stemmer_cls()

    @staticmethod
    def __safe_stemmer_by_name(stemmer_name):
        try:
            return SnowballStem.__stemmer_by_name(stemmer_name)
        except LookupError:  # pragma: no cover
            dpath = os.path.expanduser('~/nltk_data/stemmers')
            os.makedirs(dpath, exist_ok=True)
            nltk.download('snowball_data')
            return SnowballStem.__stemmer_by_name(stemmer_name)

    def __init__(self, stemmer_name, columns, drop=True, min_len=None,
                 max_len=None, **kwargs):
        self.stemmer_name = stemmer_name
        self.stemmer = SnowballStem.__safe_stemmer_by_name(stemmer_name)
        self.list_stemmer = SnowballStem._TokenListStemmer(
            stemmer=self.stemmer, min_len=min_len, max_len=max_len)
        self._columns = _interpret_columns_param(columns)
        col_str = _list_str(self._columns)
        cond_str = ''
        if min_len:
            cond_str += f' of length >= {min_len}'
        if max_len:
            if not min_len:
                cond_str += ' of length'
            cond_str += f' <= {max_len}'
        desc = SnowballStem._DEF_STEM_DESC.format(cond_str, col_str)
        super_kwargs = {
            'columns': columns,
            'value_map': self.list_stemmer,
            'drop': drop,
            'suffix': '_stem',
            'exmsg': SnowballStem._DEF_STEM_EXC_MSG.format(col_str),
            'desc': desc,
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'error'
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class DropRareTokens(ColumnsBasedPipelineStage):
    """A pipeline stage that drop rare tokens from token lists.

    Target columns must be series of token lists; i.e. every cell in the series
    is an iterable of string tokens.

    Note: The nltk package must be installed for this pipeline stage to work.

    Parameters
    ----------
    columns : single label, list-like or callable
        Column labels in the DataFrame to be transformed. Alternatively, this
        parameter can be assigned a callable returning an iterable of labels
        from an input pandas.DataFrame. See `pdpipe.cq`.
    threshold : int
        The rarity threshold to use. Only tokens appearing more than this
        number of times in a column will remain in token lists in that column.
    drop : bool, default True
        If set to True, the source columns are dropped after being transformed,
        and the resulting columns retain the names of the source columns.
        Otherwise, the new columns gain the suffix '_norare'.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[7, ['a', 'a', 'b']], [3, ['b', 'c', 'd']]]
        >>> df = pd.DataFrame(data, columns=['num', 'chars'])
        >>> rare_dropper = pdp.DropRareTokens('chars', 1)
        >>> rare_dropper(df)
           num      chars
        0    7  [a, a, b]
        1    3        [b]
    """

    def __init__(self, columns, threshold, drop=True, **kwargs):
        self._threshold = threshold
        self._drop = drop
        self._rare_removers = {}
        super_kwargs = {
            'columns': columns,
            'desc_temp': "Drop rare tokens from {}"
        }
        super_kwargs.update(**kwargs)
        super_kwargs['none_columns'] = 'error'
        super().__init__(**super_kwargs)

    class _RareRemover(object):
        def __init__(self, rare_words):
            self.rare_words = rare_words

        def __call__(self, tokens):
            return [w for w in tokens if w not in self.rare_words]

    @staticmethod
    def __get_rare_remover(series, threshold):
        token_list = [item for sublist in series for item in sublist]
        freq_dist = nltk.FreqDist(token_list)
        freq_series = pd.DataFrame.from_dict(freq_dist, orient='index')[0]
        rare_words = freq_series[freq_series <= threshold]
        return DropRareTokens._RareRemover(rare_words)

    def _fit_transform(self, df, verbose):
        inter_df = df
        columns_to_transform = self._get_columns(df, fit=True)
        if verbose:
            columns_to_transform = tqdm.tqdm(columns_to_transform)
        for colname in columns_to_transform:
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_norare"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            rare_remover = DropRareTokens.__get_rare_remover(
                source_col, self._threshold)
            self._rare_removers[colname] = rare_remover
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.map(rare_remover),
                loc=loc,
                column_name=new_name)
        self.is_fitted = True
        return inter_df

    def _transformation(self, df, verbose, fit):
        raise NotImplementedError

    def _transform(self, df, verbose):
        inter_df = df
        columns_to_transform = self._get_columns(df, fit=False)
        if verbose:
            columns_to_transform = tqdm.tqdm(columns_to_transform)
        for colname in columns_to_transform:
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_norare"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            rare_remover = self._rare_removers[colname]
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.map(rare_remover),
                loc=loc,
                column_name=new_name)
        return inter_df
