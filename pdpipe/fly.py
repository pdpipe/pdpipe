"""On the fly, relative pipeline stage creation.

Use `drop_rows_where` and `keep_rows_where` as handles to the future dataframe,
using the `[]` indexing syntax to select a (single) column to apply the logic
by, and regular binary operators such as >, >=, ==, !=, etc. to express the
condition by which rows will be kept or dropped.

For example:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
>>> df
   a   b
1  1   4
2  4   5
3  5  11
>>> pipeline = pdp.PdPipeline([
...     pdp.drop_rows_where['a'] > 4,
... ])
>>> pipeline(df)
   a  b
1  1  4
2  4  5

```

The resulting stages can be naturaly combined by logical binary operators:
& for AND, | for OR and ^ for XOR, and can also be inverted with the `~`
operator.

For example:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
>>> pipeline = pdp.PdPipeline([
...     ~ (pdp.drop_rows_where['a'] > 4),
... ])
>>> pipeline(df)
   a   b
3  5  11
>>> pipeline = pdp.PdPipeline([
...     (pdp.drop_rows_where['a'] > 3) & (pdp.drop_rows_where['b'] < 10),
... ])
>>> pipeline(df)
   a   b
1  1   4
3  5  11

```
"""

from typing import List, Set, Union

import pandas

from .core import PdPipelineStage
from . import rq


# === Auxilary pipeline stages ===

class KeepRowsByQualifier(PdPipelineStage):
    """A pipeline stage that keeps rows by a row qualifier.

    All rows which the qualifier qualifies (i.e. return a boolean series with
    True in the corresponding entries) will be kept, while all other rows will
    be dropped from input dataframes.

    Parameters
    ----------
    qualifier : RowQualifier
        An object that returns a boolean series from input dataframes. See more
        in `pdpipe.rq`.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
    >>> q = pdp.rq.ColValGt('a', 3)
    >>> pdp.fly.KeepRowsByQualifier(q).apply(df)
       a   b
    2  4   5
    3  5  11
    """

    def __init__(self, qualifier, **kwargs):
        self._keeprowsby_rq = qualifier
        super_kwargs = {
            'desc': f'Drop rows by qualifier {qualifier}',
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df: pandas.DataFrame) -> bool:
        return True

    def _transform(self, df, verbose=None):
        before_count = len(df)
        bool_ix = self._keeprowsby_rq(df)
        inter_df = df[bool_ix]
        if verbose:
            print(f"{before_count - len(inter_df)} rows dropped.")
        return inter_df

    def __and__(self, other):
        try:
            and_rq = self._keeprowsby_rq & other._keeprowsby_rq
            return type(self)(qualifier=and_rq)
        except AttributeError:
            return NotImplemented

    def __or__(self, other):
        try:
            or_rq = self._keeprowsby_rq | other._keeprowsby_rq
            return type(self)(qualifier=or_rq)
        except AttributeError:
            return NotImplemented

    def __xor__(self, other):
        try:
            xor_rq = self._keeprowsby_rq ^ other._keeprowsby_rq
            return type(self)(qualifier=xor_rq)
        except AttributeError:
            return NotImplemented

    def __invert__(self):
        not_rq = ~ self._keeprowsby_rq
        return type(self)(qualifier=not_rq)


class DropRowsByQualifier(PdPipelineStage):
    """A pipeline stage that drops rows by a row qualifier.

    All rows which the qualifier qualifies (i.e. return a boolean series with
    True in the corresponding entries) will be dropped, while all other rows
    will be kept in input dataframes.

    Parameters
    ----------
    qualifier : RowQualifier
        An object that returns a boolean series from input dataframes. See more
        in `pdpipe.rq`.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
    >>> q = pdp.rq.ColValLt('a', 3)
    >>> pdp.fly.DropRowsByQualifier(q).apply(df)
       a   b
    2  4   5
    3  5  11
    """

    def __init__(self, qualifier, **kwargs):
        self._droprowsby_rq = qualifier
        super_kwargs = {
            'desc': f'Drop rows by qualifier {qualifier}',
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df: pandas.DataFrame) -> bool:
        return True

    def _transform(self, df, verbose=None):
        before_count = len(df)
        bool_ix = ~ self._droprowsby_rq(df)
        inter_df = df[bool_ix]
        if verbose:
            print(f"{before_count - len(inter_df)} rows dropped.")
        return inter_df

    def __and__(self, other):
        try:
            and_rq = self._droprowsby_rq & other._droprowsby_rq
            return type(self)(qualifier=and_rq)
        except AttributeError:
            return NotImplemented

    def __or__(self, other):
        try:
            or_rq = self._droprowsby_rq | other._droprowsby_rq
            return type(self)(qualifier=or_rq)
        except AttributeError:
            return NotImplemented

    def __xor__(self, other):
        try:
            xor_rq = self._droprowsby_rq ^ other._droprowsby_rq
            return type(self)(qualifier=xor_rq)
        except AttributeError:
            return NotImplemented

    def __invert__(self):
        not_rq = ~ self._droprowsby_rq
        return type(self)(qualifier=not_rq)


# === Bound potential objects ===

class _DropRowsByColValColumnBoundPotential(object):

    def __init__(self, column_label: object) -> None:
        self.column_label = column_label

    # --- binary operators ---

    def __gt__(self, other):
        qualifier = rq.ColValGt(label=self.column_label, value=other)
        return DropRowsByQualifier(qualifier)

    def __ge__(self, other):
        qualifier = rq.ColValGe(label=self.column_label, value=other)
        return DropRowsByQualifier(qualifier)

    def __lt__(self, other):
        qualifier = rq.ColValLt(label=self.column_label, value=other)
        return DropRowsByQualifier(qualifier)

    def __le__(self, other):
        qualifier = rq.ColValLe(label=self.column_label, value=other)
        return DropRowsByQualifier(qualifier)

    def __eq__(self, other):
        qualifier = rq.ColValEq(label=self.column_label, value=other)
        return DropRowsByQualifier(qualifier)

    def __ne__(self, other):
        qualifier = rq.ColValNe(label=self.column_label, value=other)
        return DropRowsByQualifier(qualifier)

    # --- series methods ---

    def isin(
        self,
        value_list: Union[List[object], Set[object]],
    ) -> DropRowsByQualifier:
        q = rq.ColValIsIn(
            label=self.column_label,
            value_list=value_list,
        )
        return DropRowsByQualifier(q)

    def isna(self) -> DropRowsByQualifier:
        q = rq.ColValIsNa(self.column_label)
        return DropRowsByQualifier(q)

    def notna(self) -> DropRowsByQualifier:
        q = rq.ColValNotNa(self.column_label)
        return DropRowsByQualifier(q)


class _KeepRowsByColValColumnBoundPotential(object):

    def __init__(self, column_label: object) -> None:
        self.column_label = column_label

    # --- binary operators ---

    def __gt__(self, other):
        qualifier = rq.ColValGt(label=self.column_label, value=other)
        return KeepRowsByQualifier(qualifier)

    def __ge__(self, other):
        qualifier = rq.ColValGe(label=self.column_label, value=other)
        return KeepRowsByQualifier(qualifier)

    def __lt__(self, other):
        qualifier = rq.ColValLt(label=self.column_label, value=other)
        return KeepRowsByQualifier(qualifier)

    def __le__(self, other):
        qualifier = rq.ColValLe(label=self.column_label, value=other)
        return KeepRowsByQualifier(qualifier)

    def __eq__(self, other):
        qualifier = rq.ColValEq(label=self.column_label, value=other)
        return KeepRowsByQualifier(qualifier)

    def __ne__(self, other):
        qualifier = rq.ColValNe(label=self.column_label, value=other)
        return KeepRowsByQualifier(qualifier)

    # --- series methods ---

    def isin(
        self,
        value_list: Union[List[object], Set[object]],
    ) -> KeepRowsByQualifier:
        q = rq.ColValIsIn(
            label=self.column_label,
            value_list=value_list,
        )
        return KeepRowsByQualifier(q)

    def isna(self) -> KeepRowsByQualifier:
        q = rq.ColValIsNa(self.column_label)
        return KeepRowsByQualifier(q)

    def notna(self) -> KeepRowsByQualifier:
        q = rq.ColValNotNa(self.column_label)
        return KeepRowsByQualifier(q)


# === fly handle classes ===

class _DropRowsByColValHandle(object):

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise ValueError("pdpipe fly handles are not meant for slicing!")
        return _DropRowsByColValColumnBoundPotential(index)


class _KeepRowsByColValHandle(object):

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise ValueError("pdpipe fly handles are not meant for slicing!")
        return _KeepRowsByColValColumnBoundPotential(index)


# === fly handles ===

drop_rows_where = _DropRowsByColValHandle()

drop_rows_where.__doc__ = """
Use `drop_rows_where` as a handle to the future dataframe,
using the `[]` indexing syntax to select a (single) column to apply the logic
by, and regular binary operators such as >, >=, ==, !=, etc. to express the
condition by which rows will be kept or dropped.

For example:

>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
>>> df
   a   b
1  1   4
2  4   5
3  5  11
>>> pipeline = pdp.PdPipeline([
...     pdp.drop_rows_where['a'] > 4,
... ])
>>> pipeline(df)
   a  b
1  1  4
2  4  5

The resulting stages can be naturaly combined by logical binary operators:
& for AND, | for OR and ^ for XOR, and can also be inverted with the `~`
operator.

For example:

>>> import pandas as pd; import pdpipe as pdp;
>>> pipeline = pdp.PdPipeline([
...     ~ (pdp.drop_rows_where['a'] > 4),
... ])
>>> pipeline(df)
   a   b
3  5  11
>>> pipeline = pdp.PdPipeline([
...     (pdp.drop_rows_where['a'] > 3) & (pdp.drop_rows_where['b'] < 10),
... ])
>>> pipeline(df)
   a   b
1  1   4
3  5  11

"""

keep_rows_where = _KeepRowsByColValHandle()
