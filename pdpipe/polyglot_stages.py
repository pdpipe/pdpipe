"""Pipeline stages dependent on the polyglot Python library."""

# from polyglot.detect import Detector
#
# from pdpipe.col_generation import MapColVals
#
#
# def _safe_lang_detect(content):
#     try:
#         return Detector(some_str).languages[0].name
#     except Exception:
#         return None
#
#
# class DetectLang(MapColVals):
#     """A pipeline stage that detect language in text columns.
#
#     Parameters
#     ----------
#     columns : str or list-like
#         Column names in the DataFrame to detect language in.
#     result_columns : str or list-like, default None
#         The name of the new columns resulting from the operation. Must
#         be of the same length as columns. If None, behavior depends on the
#         drop parameter: If drop is True, the name of the source column is used;
#         otherwise, the name of the source column is used with the suffix
#         '_lang'.
#     drop : bool, default True
#         If set to True, source columns are dropped after being mapped.
#
#     Example
#     -------
#     """
#
#     def __init__(self, columns, result_columns=None, drop=True, **kwargs):
#         self._columns = _interpret_columns_param(columns, 'columns')
#         if result_columns is None:
#             if drop:
#                 self._result_columns = self._columns
#             else:
#                 self._result_columns = [col + '_lang' for col in self._columns]
#         else:
#             self._result_columns = _interpret_columns_param(
#                 result_columns, 'result_columns')
#             if len(self._result_columns) != len(self._columns):
#                 raise ValueError("columns and result_columns parameters must"
#                                  " be string lists of the same length!")
#         col_str = _list_str(self._columns)
#         sfx = 's' if len(self._columns) > 1 else ''
#         super_kwargs = {
#             'columns': self._columns,
#             'value_map': _safe_lang_detect,
#             'result_columns': self._result_columns,
#             'drop': drop,
#             'exmsg': MapColVals._DEF_MAP_COLVAL_EXC_MSG.format(sfx, col_str),
#             'appmsg': MapColVals._DEF_MAP_COLVAL_APP_MSG.format(
#                 sfx, col_str, self._value_map),
#             'desc': "Map values of column{} {} with {}.".format(
#                 sfx, col_str, self._value_map)
#         }
#         super_kwargs.update(**kwargs)
#         super().__init__(**super_kwargs)
#
