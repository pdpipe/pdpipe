"""Text processing pdpipe pipeline stages."""

import re
from typing import Optional

from pdpipe.col_generation import ApplyByCols
from pdpipe.types import ColumnsParamType, ColumnLabelsType


class RegexReplace(ApplyByCols):
    """A pipeline stage replacing regex occurences in a text column.

    Parameters
    ----------
    columns : single label, list-like or callable
        Column labels in the DataFrame which regex replacement be applied to.
        Alternatively, this parameter can be assigned a callable returning an
        iterable of labels from an input pandas.DataFrame. See `pdpipe.cq`.
    pattern : str
        The regex whose occurences will be replaced.
    replace : str
        The replacement string to use. This is equivalent to repl in re.sub.
    flags : int, default 0
        Regex flags that are compatible with Python's `re` module.
    result_columns : label or list-like of labels, default None
        The labels of the new columns resulting from the mapping operation.
        Must be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the label of the source column is
        used; otherwise, the label of the source column is casted to a string
        and concatenated with the suffix '_reg'.
    drop : bool, default True
        If set to True, source columns are dropped after being transformed.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp; import re;
    >>> data = [[4, "more than 12"], [5, "with 5 more"]]
    >>> df = pd.DataFrame(data, [1,2], ["age","text"])
    >>> clean_num = pdp.RegexReplace('text', r'\\b[0-9]+\\b', "NUM")
    >>> clean_num(df)
       age           text
    1    4  more than NUM
    2    5  with NUM more

    >>> data = [["Mr. John", 18], ["MR. Bob", 25]]
    >>> df = pd.DataFrame(data, [1,2], ["name","age"])
    >>> match_men = r'^mr.*'
    >>> censor_men = pdp.RegexReplace(
    ...    'name', match_men, "x", flags=re.IGNORECASE
    ... )
    >>> censor_men(df)
      name  age
    1    x   18
    2    x   25
    """  # noqa: W605

    class RegexReplacer(object):
        """A pickle-able regex replacement function."""

        def __init__(
            self,
            pattern_str: str,
            replace_text: str,
            flags: Optional[int] = 0,
        ) -> None:
            self.pattern_str = pattern_str
            self.replace_text = replace_text
            self.flags = flags
            self.pattern_obj = re.compile(pattern_str, flags=flags)

        def __call__(self, string: str):
            return self.pattern_obj.sub(self.replace_text, string)

    def __init__(
        self,
        columns: ColumnsParamType,
        pattern: str,
        replace: str,
        flags: Optional[int] = 0,
        result_columns: Optional[ColumnLabelsType] = None,
        drop: Optional[bool] = True,
        func_desc: Optional[str] = None,
        **kwargs,
    ):
        self._pattern_str = pattern
        self._replace = replace
        self._flags = flags
        desc_temp = "Replacing appearances of {} with '{}' in column {{}}"
        desc_temp = desc_temp.format(pattern, replace)
        super_kwargs = {
            'columns': columns,
            'func': RegexReplace.RegexReplacer(
                self._pattern_str, self._replace, self._flags),
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
    columns : single label, list-like or callable
        Names of token list columns on which to apply token filtering.
        Alternatively, this parameter can be assigned a callable returning an
        iterable of labels from an input pandas.DataFrame. See `pdpipe.cq`.
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
    columns : single label, list-like or callable
        Names of token list columns on which to apply token filtering.
        Alternatively, this parameter can be assigned a callable returning an
        iterable of labels from an input pandas.DataFrame. See `pdpipe.cq`.
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
