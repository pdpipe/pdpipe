"""Whitelist and blacklist of pandas.Series transforms."""


SERIES_TRANSFORMS_WHITELIST = [
    'add', 'add_prefix', 'add_suffix', 'append', 'argsort', 'asof', 'backfill', 'combine_first', 'convert_dtypes', 'copy', 'describe', 'diff', 'div', 'divide', 'drop', 'drop_duplicates', 'dropna', 'duplicated', 'eq', 'explode', 'fillna', 'floordiv', 'ge', 'gt', 'isin', 'isna', 'isnull', 'le', 'lt', 'map', 'mod', 'mode', 'mul', 'multiply', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 'pad', 'pct_change', 'pow', 'radd', 'rdiv', 'reindex', 'reindex_like', 'rename', 'rename_axis', 'replace', 'rfloordiv', 'rmod', 'rmul', 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'set_axis', 'set_flags', 'shift', 'sort_values', 'squeeze', 'sub', 'subtract', 'truediv', 'value_counts', 'view',  # noqa: E501
    'reset_index', 'combine', 'at_time', 'compare',
    'apply', 'between', 'between_time', 'swaplevel', 'droplevel',
    'dot', 'transform', 'tshift', 'tz_localize',
    'to_period', 'to_timestamp',
    'first', 'kurt', 'kurtosis', 'last', 'mad', 'max', 'mean', 'median', 'min', 'prod', 'product', 'quantile', 'sem', 'skew', 'std', 'sum', 'var', 'xs',  # noqa: E501
]

SERIES_TRANSFORMS_BLACKLIST = [
    'agg', 'aggregate', 'groupby', 'align',
    'count', 'divmod', 'rdivmod',
]
