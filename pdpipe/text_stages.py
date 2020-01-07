"""Text processing pdpipe pipeline stages."""

import re

from pdpipe.col_generation import ApplyByCols
from pdpipe.shared import (
    _list_str
)


class RegexReplace(ApplyByCols):
    """A pipeline stage replacing regex occurences in a text column.

    Parameters
    ----------
    columns : str or list-like
        Names of columns on which to apply regex replacement.
    pattern : str
        The regex whose occurences will be replaced.
    replace : str
        The replacement string to use.
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

    _BASE_STR = "Replacing appearances of {} with '{}' in column{} {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_APP_MSG_SUFFIX = "..."
    _DEF_DESCRIPTION_SUFFIX = "."

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
        col_str = _list_str(columns)
        sfx = "s" if len(columns) > 1 else ""
        base_str = RegexReplace._BASE_STR.format(
            pattern, replace, sfx, col_str)
        super_kwargs = {
            'columns': columns,
            'func': lambda x: self._pattern_obj.sub(self._replace, x),
            'colbl_sfx': '_regex',
            'drop': drop,
            'exmsg': base_str + ApplyByCols._DEF_EXC_MSG_SUFFIX,
            'appmsg': base_str + ApplyByCols._DEF_APP_MSG_SUFFIX,
            'desc': base_str + ApplyByCols._DEF_DESCRIPTION_SUFFIX,
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

    _BASE_STR = "Filtering out tokens of length{} with in column{} {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_APP_MSG_SUFFIX = "..."
    _DEF_DESCRIPTION_SUFFIX = "."

    def __init__(
        self, columns, min_len, max_len=None, result_columns=None, drop=True,
        **kwargs
    ):
        self._min_len = min_len
        self._max_len = max_len
        col_str = _list_str(columns)
        sfx = "s" if len(columns) > 1 else ""
        cond_str = " > {}".format(min_len)
        if max_len:
            cond_str += " < {}".format(max_len)
        base_str = DropTokensByLength._BASE_STR.format(cond_str, sfx, col_str)

        def _token_filter(token_list):
            return [x for x in token_list if len(x) >= min_len]

        if max_len:

            def _token_filter(token_list):  # noqa: F811
                return [
                    x for x in token_list
                    if len(x) >= min_len and len(x) <= max_len
                ]

        super_kwargs = {
            "columns": columns,
            "func": _token_filter,
            "colbl_sfx": "_filtered",
            "drop": drop,
            "exmsg": base_str + DropTokensByLength._DEF_EXC_MSG_SUFFIX,
            "appmsg": base_str + DropTokensByLength._DEF_APP_MSG_SUFFIX,
            "desc": base_str + DropTokensByLength._DEF_DESCRIPTION_SUFFIX,
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

    _BASE_STR = "Filtering out tokens{} in column{} {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_APP_MSG_SUFFIX = "..."
    _DEF_DESCRIPTION_SUFFIX = "."

    def __init__(
        self, columns, bad_tokens, result_columns=None, drop=True,
        **kwargs
    ):
        self._bad_tokens = bad_tokens
        col_str = _list_str(columns)
        sfx = "s" if len(columns) > 1 else ""
        cond_str = ""
        if len(bad_tokens) < 10:
            cond_str = "in list [" + " ".join(bad_tokens) + "]"
        base_str = DropTokensByList._BASE_STR.format(cond_str, sfx, col_str)

        def _token_filter(token_list):
            return [x for x in token_list if x not in bad_tokens]

        super_kwargs = {
            "columns": columns,
            "func": _token_filter,
            "colbl_sfx": "_filtered",
            "drop": drop,
            "exmsg": base_str + DropTokensByLength._DEF_EXC_MSG_SUFFIX,
            "appmsg": base_str + DropTokensByLength._DEF_APP_MSG_SUFFIX,
            "desc": base_str + DropTokensByLength._DEF_DESCRIPTION_SUFFIX,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)
