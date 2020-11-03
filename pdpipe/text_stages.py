"""Text processing pdpipe pipeline stages."""

import re

from pdpipe.col_generation import ApplyByCols


class RegexReplace(ApplyByCols):
    """A pipeline stage replacing regex occurences in a text column.

    Parameters
    ----------
    columns : str or list-like
        Names of columns on which to apply regex replacement.
    pattern : str
        The regex whose occurences will be replaced.
    replace : str
        The replacement string to use. This is equivalent to repl in re.sub.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the name of the source column is used;
        otherwise, the name of the source column is used with the suffix
        '_reg'.
    drop : bool, default True
        If set to True, source columns are dropped after being transformed.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[4, "more than 12"], [5, "with 5 more"]]
        >>> df = pd.DataFrame(data, [1,2], ["age","text"])
        >>> clean_num = pdp.RegexReplace('text', r'\\b[0-9]+\\b', "NUM")
        >>> clean_num(df)
           age           text
        1    4  more than NUM
        2    5  with NUM more
    """  # noqa: W605

    class RegexReplacer(object):
        """A pickle-able regex replacement function."""

        def __init__(self, pattern_obj, replace_text):
            self.pattern_obj = pattern_obj
            self.replace_text = replace_text

        def __call__(self, x):
            return self.pattern_obj.sub(self.replace_text, x)

    def __init__(
        self,
        columns,
        pattern,
        replace,
        result_columns=None,
        drop=True,
        func_desc=None,
        **kwargs
    ):
        self._pattern = pattern
        self._replace = replace
        self._pattern_obj = re.compile(pattern)
        desc_temp = "Replacing appearances of {} with '{}' in column {{}}"
        desc_temp = desc_temp.format(pattern, replace)
        super_kwargs = {
            'columns': columns,
            'func': RegexReplace.RegexReplacer(
                self._pattern_obj, self._replace),
            'suffix': '_regex',
            'result_columns': result_columns,
            'drop': drop,
            'desc_temp': desc_temp,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)


class DropTokensByLength(ApplyByCols):
    """A pipeline stage removing tokens by length in string-token list columns.

    Parameters
    ----------
    columns : str or list-like
        Names of token list columns on which to apply token filtering.
    min_len : int
        The minimum length of tokens to keep. Tokens of shorter length are
        removed from all token lists.
    max_len : int, default None
        The maximum length of tokens to keep. If provided, tokens of longer
        length are removed from all token lists.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation.
        Must be of the same length as columns. If None, behavior depends on
        the drop parameter: If drop is True, the name of the source column
        is used; otherwise, the name of the source column is used with the
        suffix '_filtered'.
    drop : bool, default True
        If set to True, source columns are dropped after being transformed.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[4, ["a", "bad", "nice"]], [5, ["good", "university"]]]
        >>> df = pd.DataFrame(data, [1,2], ["age","text"])
        >>> filter_tokens = pdp.DropTokensByLength('text', 3, 5)
        >>> filter_tokens(df)
           age         text
        1    4  [bad, nice]
        2    5       [good]
    """  # noqa: W605

    class MinLengthTokenFilter(object):

        def __init__(self, min_len):
            self.min_len = min_len

        def __call__(self, token_list):
            return [x for x in token_list if len(x) >= self.min_len]

    class MinMaxLengthTokenFilter(object):

        def __init__(self, min_len, max_len):
            self.min_len = min_len
            self.max_len = max_len

        def __call__(self, token_list):
            return [
                x for x in token_list
                if len(x) >= self.min_len and len(x) <= self.max_len
            ]

    def __init__(
        self, columns, min_len, max_len=None, result_columns=None, drop=True,
        **kwargs
    ):
        self._min_len = min_len
        self._max_len = max_len
        token_filter = DropTokensByLength.MinLengthTokenFilter(min_len)
        cond_str = f" > {min_len}"
        if max_len:
            token_filter = DropTokensByLength.MinMaxLengthTokenFilter(
                min_len=min_len, max_len=max_len)
            cond_str += f" < {max_len}"
        desc_temp = "Filtering out tokens of length{} in columns {{}}"
        desc_temp = desc_temp.format(cond_str)
        super_kwargs = {
            'columns': columns,
            'func': token_filter,
            'result_columns': result_columns,
            'drop': drop,
            'suffix': "_filtered",
            'desc_temp': desc_temp,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)


class DropTokensByList(ApplyByCols):
    """A pipeline stage removing specific tokens in string-token list columns.

    Parameters
    ----------
    columns : str or list-like
        Names of token list columns on which to apply token filtering.
    bad_tokens : list of str
        The list of string tokens to remove from all token lists.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation.
        Must be of the same length as columns. If None, behavior depends on
        the drop parameter: If drop is True, the name of the source column
        is used; otherwise, the name of the source column is used with the
        suffix '_filtered'.
    drop : bool, default True
        If set to True, source columns are dropped after being transformed.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[4, ["a", "bad", "cat"]], [5, ["bad", "not", "good"]]]
        >>> df = pd.DataFrame(data, [1,2], ["age","text"])
        >>> filter_tokens = pdp.DropTokensByList('text', ['bad'])
        >>> filter_tokens(df)
           age         text
        1    4     [a, cat]
        2    5  [not, good]
    """  # noqa: W605

    class ListTokenFilter(object):

        def __init__(self, bad_tokens):
            self.bad_tokens = bad_tokens

        def __call__(self, token_list):
            return [x for x in token_list if x not in self.bad_tokens]

    def __init__(
        self, columns, bad_tokens, result_columns=None, drop=True,
        **kwargs
    ):
        self._bad_tokens = bad_tokens
        cond_str = ""
        if len(bad_tokens) < 10:
            cond_str = " in list [" + " ".join(bad_tokens) + "]"
        base_str = "Filtering out tokens{} in columns {{}}"
        desc_temp = base_str.format(cond_str)
        super_kwargs = {
            'columns': columns,
            'func': DropTokensByList.ListTokenFilter(bad_tokens),
            'result_columns': result_columns,
            'drop': drop,
            'suffix': "_filtered",
            'desc_temp': desc_temp,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)
