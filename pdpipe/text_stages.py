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


# class DropTokensByLength(ApplyByCols):
#     """A pipeline stage replacing regex occurences in a text column.
#
#     Parameters
#     ----------
#     columns : str or list-like
#         Names of columns on which to apply regex replacement.
#     pattern : str
#         The regex whose occurences will be replaced.
#     replace : str
#         The replacement string to use.
#     result_columns : str or list-like, default None
#         The names of the new columns resulting from the mapping operation.
#         Must be of the same length as columns. If None, behavior depends on
#         the drop parameter: If drop is True, the name of the source column
#         is used; otherwise, the name of the source column is used with the
#         suffix '_reg'.
#     drop : bool, default True
#         If set to True, source columns are dropped after being transformed.
#
#     Example
#     -------
#         >>> import pandas as pd; import pdpipe as pdp;
#         >>> data = [[4, "more than 12"], [5, "with 5 more"]]
#         >>> df = pd.DataFrame(data, [1,2], ["age","text"])
#         >>> clean_num = pdp.RegexReplace('text', r'\\b[0-9]+\\b', "NUM")
#         >>> clean_num(df)
#            age           text
#         1    4  more than NUM
#         2    5  with NUM more
#     """  # noqa: W605
#
#     _BASE_STR = "Replacing appearances of {} with '{}' in column{} {}"
#     _DEF_EXC_MSG_SUFFIX = " failed."
#     _DEF_APP_MSG_SUFFIX = "..."
#     _DEF_DESCRIPTION_SUFFIX = "."
#
#     def __init__(
#         self,
#         columns,
#         pattern,
#         replace,
#         result_columns=None,
#         drop=True,
#         func_desc=None,
#         **kwargs
#     ):
#         self._pattern = pattern
#         self._replace = replace
#         self._pattern_obj = re.compile(pattern)
#         col_str = _list_str(columns)
#         sfx = "s" if len(columns) > 1 else ""
#         base_str = RegexReplace._BASE_STR.format(
#             pattern, replace, sfx, col_str)
#         super_kwargs = {
#             'columns': columns,
#             'func': lambda x: self._pattern_obj.sub(self._replace, x),
#             'colbl_sfx': '_regex',
#             'drop': drop,
#             'exmsg': base_str + ApplyByCols._DEF_EXC_MSG_SUFFIX,
#             'appmsg': base_str + ApplyByCols._DEF_APP_MSG_SUFFIX,
#             'desc': base_str + ApplyByCols._DEF_DESCRIPTION_SUFFIX,
#         }
#         super_kwargs.update(**kwargs)
#         super().__init__(**super_kwargs)
#
#
#
#
#
#
