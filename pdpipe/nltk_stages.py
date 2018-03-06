"""Pipeline stages dependent on the scikit-learn Python library."""

import importlib

import nltk
import pandas as pd

from pdpipe.core import PipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.col_generation import MapColVals
from pdpipe.shared import (
    _interpret_columns_param,
    _list_str
)


class TokenizeWords(MapColVals):
    """A pipeline stage that tokenize a sentence into words by whitespaces.

    Parameters
    ----------
    columns : str or list-like
        Column names in the DataFrame to be tokenized.
    drop : bool, default True
        If set to True, the source columns are dropped after being tokenized,
        and the resulting tokenized columns retain the names of the source
        columns. Otherwise, tokenized columns gain the suffix '_tok'.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[3.2, "Kick the baby!"]], [1], ['freq', 'content'])
    >>> tokenize_stage = pdp.TokenizeWords('content')
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
            nltk.download('punkt')

    def __init__(self, columns, drop=True, **kwargs):
        self.__check_punkt()
        self._columns = _interpret_columns_param(columns, 'columns')
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': nltk.word_tokenize,
            'drop': drop,
            'suffix': '_tok',
            'exmsg': TokenizeWords._DEF_TOKENIZE_EXC_MSG.format(col_str),
            'appmsg': TokenizeWords._DEF_TOKENIZE_APP_MSG.format(col_str),
            'desc': "Tokenize {}".format(col_str),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class UntokenizeWords(MapColVals):
    """A pipeline stage that joins token lists to whitespace-seperated strings.

    Parameters
    ----------
    columns : str or list-like
        Column names in the DataFrame to be untokenized.
    drop : bool, default True
        If set to True, the source columns are dropped after being untokenized,
        and the resulting columns retain the names of the source columns.
        Otherwise, untokenized columns gain the suffix '_untok'.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, ['Shake', 'and', 'bake!']]]
    >>> df = pd.DataFrame(data, [1], ['freq', 'content'])
    >>> untokenize_stage = pdp.UntokenizeWords('content')
    >>> untokenize_stage(df)
       freq          content
    1   3.2  Shake and bake!
    """

    _DEF_UNTOKENIZE_EXC_MSG = ("Unokenize stage failed because not all columns"
                               " {} are present in input dataframe and are of"
                               " dtype object.")

    @staticmethod
    def _untokenize_list(token_list):
        return ' '.join(token_list)

    def __init__(self, columns, drop=True, **kwargs):
        self._columns = _interpret_columns_param(columns, 'columns')
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': UntokenizeWords._untokenize_list,
            'drop': drop,
            'suffix': '_untok',
            'exmsg': UntokenizeWords._DEF_UNTOKENIZE_EXC_MSG.format(col_str),
            'appmsg': "Untokenizing {}".format(col_str),
            'desc': "Untokenize {}".format(col_str),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class RemoveStopwords(MapColVals):
    """A pipeline stage that removes stopwords from a tokenized list.

    Parameters
    ----------
    langugae : str
        The language of the stopwords. Should be one of the languages
        supported by the NLTK Stopwords Corpus.
    columns : str or list-like
        Column names in the DataFrame from which to remove stopwords.
    drop : bool, default True
        If set to True, the source columns are dropped after stopword removal,
        and the resulting columns retain the names of the source columns.
        Otherwise, resulting columns gain the suffix '_nostop'.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, ['kick', 'the', 'baby']]]
    >>> df = pd.DataFrame(data, [1], ['freq', 'content'])
    >>> remove_stopwords = pdp.RemoveStopwords('english', 'content')
    >>> remove_stopwords(df)
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
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            return stopwords.words(language)

    def __init__(self, language, columns, drop=True, **kwargs):
        self._language = language
        self._stopwords_list = RemoveStopwords.__stopwords_by_language(
            language)
        self._stopwords_remover = RemoveStopwords._StopwordsRemover(
            self._stopwords_list)
        self._columns = _interpret_columns_param(columns, 'columns')
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': self._stopwords_remover,
            'drop': drop,
            'suffix': '_nostop',
            'exmsg': RemoveStopwords._DEF_STOPWORDS_EXC_MSG.format(col_str),
            'appmsg': RemoveStopwords._DEF_STOPWORDS_APP_MSG.format(col_str),
            'desc': "Removing stopwords from {}".format(col_str),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class SnowballStem(MapColVals):
    """A pipeline stage that stems words in a list using the Snowball stemmer.

    Parameters
    ----------
    stemmer_name : str
        The name of the Snowball stemmer to use. Should be one of the Snowball
        stemmers implemented by nltk. E.g. 'EnglishStemmer'.
    columns : str or list-like
        Column names in the DataFrame to stem tokens in.
    drop : bool, default True
        If set to True, the source columns are dropped after stemming, and the
        resulting columns retain the names of the source columns. Otherwise,
        resulting columns gain the suffix '_stem'.

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
    _DEF_STEM_APP_MSG = "Stemming tokens in {}..."

    class _TokenListStemmer(object):
        def __init__(self, stemmer):
            self.stemmer = stemmer

        def __call__(self, token_list):
            return [self.stemmer.stem(w) for w in token_list]

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
            nltk.download('snowball_data')
            return SnowballStem.__stemmer_by_name(stemmer_name)

    def __init__(self, stemmer_name, columns, drop=True, **kwargs):
        self.stemmer_name = stemmer_name
        self.stemmer = SnowballStem.__safe_stemmer_by_name(stemmer_name)
        self.list_stemmer = SnowballStem._TokenListStemmer(self.stemmer)
        self._columns = _interpret_columns_param(columns, 'columns')
        col_str = _list_str(self._columns)
        super_kwargs = {
            'columns': columns,
            'value_map': self.list_stemmer,
            'drop': drop,
            'suffix': '_stem',
            'exmsg': SnowballStem._DEF_STEM_EXC_MSG.format(col_str),
            'appmsg': SnowballStem._DEF_STEM_APP_MSG.format(col_str),
            'desc': "Stem tokens in {}".format(col_str),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return super()._prec(df) and all(
            col_type == object for col_type in df.dtypes[self._columns])


class DropRareTokens(PipelineStage):
    """A pipeline stage that drop rare tokens from token lists.

    Parameters
    ----------
    columns : str or list-like
        Column names in the DataFrame for which to drop rare words.
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

    _DEF_RARE_EXC_MSG = ("DropRareTokens stage failed because not all columns "
                         "{} were found in input dataframe.")

    def __init__(self, columns, threshold, drop=True, **kwargs):
        self._columns = _interpret_columns_param(columns, 'columns')
        self._threshold = threshold
        self._drop = drop
        self._rare_removers = {}
        col_str = _list_str(self._columns)
        super_kwargs = {
            'exmsg': DropRareTokens._DEF_RARE_EXC_MSG.format(col_str),
            'appmsg': "Dropping rare tokens from {}...".format(col_str),
            'desc': "Drop rare tokens from {}".format(col_str)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns).issubset(df.columns)

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

    def _op(self, df, verbose):
        inter_df = df
        for colname in self._columns:
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

    def _transform(self, df, verbose):
        inter_df = df
        for colname in self._columns:
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
