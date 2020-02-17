"""Column qualifiers for pdpipe.

Most pipeline stages in pdpipe can accept three types of variables as arguments
for the `columns` parameter of their constructor: a single column label, a list
of column labels, or a callable. The first is interpreted as the label of the
single column on which the pipeline stage should operate and the second as a
list of such labels, while the a callable is assumed to determine dynamically
what columns should the stage be applied to. It is thus supplied with the
entire input dataframe, and is expected to return a list of column labels. This
is true for every application of the pipeline stage, both in fit time and in
any future transform.

A naive callable, such as `lambda df: [lbl for lbl in df.columns if lbl[0] ==
'a']`, meant to cause the pipeline stage to operate on any column with a label
starting with the letter 'a', might result in unexpected errors: if after the
pipeline was fitted it gets ― in transform time ― a dataframe with new columns
starting with 'a', it is will transform them as well, which will in turn might
(1) lead to unexpected errors, as the newer columns might not be valid input to
that stage, and (2) lead to a change in schema, which might cause errors down
the pipeline, especially if there's a fitted machine learning model down the
pipeline.

Of course, this might be the desired behaviour ― to tranform coloumns 'alec'
and 'alex' on the first `apply` call and 'alec' and 'apoxy' in transform time ―
but usually this is not the case. In fact, in common machine learning
scenarios ― whether it is fitting pre-processing parameters on the train set
and using the resulting pipeline to transform the test and validation sets, or
fitting the pre-processing pipeline on current data and deploying it to
transform incoming data in production ― we would exepct this criterion to be
applied once, when the pipeline is being fit, and for future calls for it to
only transform the 'alec' and 'alex' columns, ignoring any other columns
starting with 'a' that are newly-encountered in transform time, and also
explicitly fails if either once of the two columns, 'alec' and 'alex', is
missing. This kind of behaviour is the only way to ensure preservation of both
form and semantics of input vectors to our models down the pipeline.

To enable this more sophicticated behaviour this module - `pdpipe.cq` - exposes
a way to easily generate `ColumnQualifier` objects, which are callables that do
exactly what was described above: Apply some criteria to determine a set of
input columns when a pipeline is being fitted, but fixing it afterwards, on
future calls.

Practically, this objects all expose `fit`, `transform` and `fit_transform`
methods, and while the first time they are called the `fit_transform` method is
called, future calls will actually call the `transform` method. Also, since
they already expose this more poweful API, pipeline stages use it to enable
an even more powerful (and quite frankly, expected) behavior: When a pipeline's
`fit_transfrom` or `fit` methods are called, it calls the `fit_transform`
method of the column qualifier it uses, so the qualifier itself is refitted
even if it is already fit. Naturally, if the callable has no `fit_transform`
method the code gracefully backs-off to just applying it, which allows the use
of unfittable functions and lambdas as column qualifiers as well.

Note that any callable can be wrapped in a ColumnQualifier object to achieve
this fittable behaviour. For example, to get a pipeline stage that drops all
columns of data type `numpy.int64`, but also "remembers" that list after fit:

```python
pipeline += pdp.ColDrop(columns=pdp.cq.ColumnQualifier(lambda df: [
  l for l, s in df.iteritems()
  if s.dtype == np.int64
]))
```

ColumnQualifier objects also support the &, ^ and | binary boolean operators -
representing boolean and, xor and or, respectively - and the ~ unary boolean
operator - representing the boolean not operator. Finally, the - binary
operator is implemented to represent the NOT IN non-symetric binary relation
between two qualifiers (the difference operator on the resulting sets).

So for example, to get a qualifier that qualifies all columns that have AT
LEAST two missing values, one can use:

    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[None, 1, 2],[None, None, 5]], [1,2], ['ph', 'grade', 'age'])
    >>> cq = ~ pdp.cq.WithAtMostMissingValues(1)
    >>> cq(df)
    ['ph']

While to get a qualifier matching all columns with at most one missing value
AND starting with 'gr' one can use:

    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[None, 1, 2],[None, None, 5]], [1,2], ['grep', 'grade', 'age'])
    >>> cq = pdp.cq.WithAtMostMissingValues(1) & pdp.cq.StartWith('gr')
    >>> cq(df)
    ['grade']

And a qualifier that qualifies all columns with no missing values except those
that start with 'b' can be generated with:

    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[1, 2, 3, 4],[5, 6, 7, None]], [1,2], ['abe', 'bee', 'cry', 'no'])
    >>> cq = pdp.cq.WithoutMissingValues() - pdp.cq.StartWith('b')
    >>> cq(df)
    ['abe', 'cry']
"""

from .shared import _list_str


class UnfittedColumnQualifierError(Exception):
    """An exception raised when a (non-fit) transform is attempted with an
    unfitted column qualifier.
    """


class ColumnQualifier(object):
    """A fittable qualifier that returns column labels from an input dataframe.

    Parameters
    ----------
    func : callable
        A callable that given an input pandas.DataFrame objects returns a list
        of labels of a subset of the columns of the input dataframe.
    fittable : bool, default True
        If set to false, this qualifier becomes unfittable, and `func` is
        called on every call to transform. True by default.
    subset : bool, default False
        If set to true, fitted qualifiers return the subset of fitted columns
        found in input dataframes during transform, in the order they appeared
        when fitted (NOT in the order they appear in the input dataframe of the
        transform). False by default, which means fitted qualifiers return the
        FULL list of fitted columns, ignoring input dataframes completely on
        transforms. When combined with most pipeline stages, this means the
        stage will fail on its precondition if trying to transform with it a
        dataframe that is missing some values in the fitted qualifier.

    Example
    -------
        >>> import numpy as np; import pdpipe as pdp;
        >>> cq = pdp.cq.ColumnQualifier(lambda df: [
        ...    l for l, s in df.iteritems()
        ...    if s.dtype == np.int64 and l in ['a', 'b', 5]
        ... ])
        >>> cq
        <ColumnQualifier: Qualify columns by function>
        >>> col_drop = pdp.ColDrop(columns=cq)
    """

    def __init__(self, func, fittable=None, subset=None):
        if fittable is None:
            fittable = True
        self._cqfunc = func
        self._fittable = fittable
        self._subset = subset

    def __call__(self, df):
        """Returns column labels of qualified columns from an input dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe, from which columns are selected.

        Returns
        -------
        list of objects
            A list of labels of the qualified columns for the input dataframe.
        """
        try:
            return self.transform(df)
        except UnfittedColumnQualifierError:
            return self.fit_transform(df)

    def fit_transform(self, df):
        """Fits this qualifier and returns the labels of the qualifying columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe, from which columns are selected.

        Returns
        -------
        list of objects
            A list of labels of the qualified columns for the input dataframe.
        """
        self._columns = self._cqfunc(df)
        return self._columns

    def fit(self, df):
        """Fits this qualifier on the input dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe, from which columns are selected.

        """
        self.fit_transform(df)

    def transform(self, df):
        """Applies and returns the labels of the qualifying columns.

        Is this ColumnQualifier is fittable, it will return the list of column
        labels that was determined when fitted (or the subset of it that can
        be found in the input datarame), if it's fitted, and throw an exception
        if it is not.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe, from which columns are selected.

        Returns
        -------
        list of objects
            A list of labels of the qualified columns for the input dataframe.
        """
        if not self._fittable:
            return self._cqfunc(df)
        try:
            if self._subset:
                return [x for x in self._columns if x in df.columns]
            return self._columns
        except AttributeError:
            raise UnfittedColumnQualifierError

    def __repr__(self):
        fstr = ''
        if self._cqfunc.__doc__:  # pragma: no cover
            fstr = ' - {}'.format(self._cqfunc.__doc__)
        return "<ColumnQualifier: Qualify columns by function{}>".format(fstr)

    # --- overriding boolean operators ---

    @staticmethod
    def _x_inorderof_y(x, y):
        return [i for i in y if i in x]

    def __and__(self, other):
        try:
            ofunc = other._cqfunc
            def _cqfunc(df):  # noqa: E306
                return ColumnQualifier._x_inorderof_y(
                    x=set(self._cqfunc(df)).intersection(ofunc(df)),
                    y=df.columns,
                )
            _cqfunc.__doc__ = '{} AND {}'.format(
                self._cqfunc.__doc__ or 'Anonymous qualifier 1',
                other._cqfunc.__doc__ or 'Anonymous qualifier 2',
            )
            return ColumnQualifier(func=_cqfunc)
        except AttributeError:
            return NotImplemented

    def __xor__(self, other):
        try:
            ofunc = other._cqfunc
            def _cqfunc(df):  # noqa: E306
                return ColumnQualifier._x_inorderof_y(
                    x=set(self._cqfunc(df)).symmetric_difference(ofunc(df)),
                    y=df.columns,
                )
            _cqfunc.__doc__ = '{} XOR {}'.format(
                self._cqfunc.__doc__ or 'Anonymous qualifier 1',
                other._cqfunc.__doc__ or 'Anonymous qualifier 2',
            )
            return ColumnQualifier(func=_cqfunc)
        except AttributeError:
            return NotImplemented

    def __or__(self, other):
        try:
            ofunc = other._cqfunc
            def _cqfunc(df):  # noqa: E306
                return ColumnQualifier._x_inorderof_y(
                    x=set(self._cqfunc(df)).union(ofunc(df)),
                    y=df.columns,
                )
            _cqfunc.__doc__ = '{} OR {}'.format(
                self._cqfunc.__doc__ or 'Anonymous qualifier 1',
                other._cqfunc.__doc__ or 'Anonymous qualifier 2',
            )
            return ColumnQualifier(func=_cqfunc)
        except AttributeError:
            return NotImplemented

    def __sub__(self, other):
        try:
            ofunc = other._cqfunc
            def _cqfunc(df):  # noqa: E306
                return ColumnQualifier._x_inorderof_y(
                    x=set(self._cqfunc(df)).difference(ofunc(df)),
                    y=df.columns,
                )
            _cqfunc.__doc__ = '{} NOT IN {}'.format(
                self._cqfunc.__doc__ or 'Anonymous qualifier 1',
                other._cqfunc.__doc__ or 'Anonymous qualifier 2',
            )
            return ColumnQualifier(func=_cqfunc)
        except AttributeError:
            return NotImplemented

    def __invert__(self):
        def _cqfunc(df):  # noqa: E306
            return ColumnQualifier._x_inorderof_y(
                x=set(df.columns).difference(self._cqfunc(df)),
                y=df.columns,
            )
        _cqfunc.__doc__ = 'NOT {}'.format(
            self._cqfunc.__doc__ or 'Anonymous qualifier'
        )
        return ColumnQualifier(func=_cqfunc)


def is_fittable_column_qualifier(obj):
    """Returns True for objects that are fittable ColumnQualifier objects.

    Parameters
    ----------
    obj : object
        The object to examine.

    Returns
    -------
    bool
        True if the given object is an instance of ColumnQualifier and
        fittable, False otherwise.
    """
    return isinstance(obj, ColumnQualifier) and obj._fittable


class AllColumns(ColumnQualifier):
    """Selectes all columns in input dataframes..

    Parameters
    ----------
    **kwargs
        Accepts all keyword arguments of the constructor of
        ColumnQualifier. See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[8,1],[5,2]], [1,2], ['a', 'b'])
        >>> cq = pdp.cq.AllColumns()
        >>> cq
        <ColumnQualifier: Qualify all columns>
        >>> cq(df)
        ['a', 'b']
        >>> df2 = pd.DataFrame([[8,1],[5,2]], [1,2], ['b', 'c'])
        >>> cq(df2)
        ['a', 'b']
        >>> cq = pdp.cq.AllColumns(fittable=False)
        >>> cq(df)
        ['a', 'b']
        >>> cq(df2)
        ['b', 'c']
        >>> cq = pdp.cq.AllColumns(subset=True)
        >>> cq(df)
        ['a', 'b']
        >>> cq(df2)
        ['b']
    """

    def __init__(self, **kwargs):
        def _cqfunc(df):  # noqa: E306
            return list(df.columns)
        kwargs['func'] = _cqfunc
        super().__init__(**kwargs)

    def __repr__(self):
        return "<ColumnQualifier: Qualify all columns>"


class ByColumnCondition(ColumnQualifier):
    """A fittable column qualifier based on a per-column condition.

    Parameters
    ----------
    cond : callable
        A callaable that given an input pandas.Series object returns a boolean
        value.
    safe : bool, default False
        If set to True, every call to given condition `cond` is is wrapped in
        a way that interprets every raised exception as a returned False value.
        This is useful when generating qualifiers based on conditions that
        assume a specific datatype for the checked column.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        ColumnQualifier. See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame(
        ...    [[1, 2, 'A'],[4, 1, 'C']], [1,2], ['age', 'count', 'grade'])
        >>> cq = pdp.cq.ByColumnCondition(lambda s: s.sum() > 3, safe=True)
        >>> cq(df)
        ['age']
    """

    def __init__(self, cond, safe=False, **kwargs):
        self._cond = cond
        if safe:
            def _safe_cond(series):
                try:
                    return cond(series)
                except Exception:
                    return False
            self._cond = _safe_cond
        def _cqfunc(df):  # noqa: E306
            return list([
                lbl for lbl, series in df.iteritems()
                if self._cond(series)
            ])
        kwargs['func'] = _cqfunc
        super().__init__(**kwargs)


class ByLabels(ColumnQualifier):
    """Selectes all columns with the given label or labels.

    Parameters
    ----------
    labels : single label or list-like
        Columns labels which qualify.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        ColumnQualifier. See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame(
        ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
        >>> cq = pdp.cq.ByLabels('num')
        >>> cq(df)
        ['num']
        >>> cq = pdp.cq.ByLabels(['chr', 'nur'])
        >>> cq(df)
        ['chr', 'nur']
        >>> cq = pdp.cq.ByLabels(['num', 'foo'])
        >>> cq(df)
        ['num']
    """

    def __init__(self, labels, **kwargs):
        if isinstance(labels, str) or not hasattr(labels, '__iter__'):
            labels = [labels]
        self._labels = labels
        self._labels_str = _list_str(self._labels)
        def _cqfunc(df):  # noqa: E306
            return [
                lbl for lbl in df.columns
                if lbl in self._labels
            ]
        _cqfunc.__doc__ = "Columns  wwith labels in {}".format(
            self._labels_str)
        kwargs['func'] = _cqfunc
        super().__init__(**kwargs)

    def __repr__(self):
        return "<ColumnQualifier: By labels in {}>".format(self._labels_str)


def columns_to_qualifier(columns):
    """Converts the given columns parameter to an equivalent column qualifier.

    Parameters
    ----------
    columns : single label, list-like or callable
        The label, or an iterable of labels, of columns. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See pdpipe.cq.

    Returns
    -------
    qualifier : ColumnQualifier
        The equivalent ColumnQualifier object.

    Example
    -------
        >>> import pdpipe as pdp;
        >>> pdp.cq.columns_to_qualifier('nu')
        <ColumnQualifier: By labels in nu>
        >>> pdp.cq.columns_to_qualifier(['nu', 'bu'])
        <ColumnQualifier: By labels in nu, bu>
        >>> pdp.cq.columns_to_qualifier(lambda df: [l for l in df.columns])
        <ColumnQualifier: Qualify columns by function>
    """
    if callable(columns):
        if isinstance(columns, ColumnQualifier):
            return columns
        return ColumnQualifier(columns, fittable=False)
    return ByLabels(columns)


class StartWith(ColumnQualifier):
    """Selectes all columns that start with the given string.

    Parameters
    ----------
    prefix : str
        The prefix which qualifies columns.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        ColumnQualifier. See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame(
        ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
        >>> cq = pdp.cq.StartWith('nu')
        >>> cq
        <ColumnQualifier: Columns starting with nu>
        >>> cq(df)
        ['num', 'nur']
    """

    @staticmethod
    def _safe_startwith(string, prefix):
        try:
            return string.startswith(prefix)
        except AttributeError:
            return False

    def __init__(self, prefix, **kwargs):
        self._prefix = prefix
        def _cqfunc(df):  # noqa: E306
            return [
                lbl for lbl in df.columns
                if StartWith._safe_startwith(lbl, self._prefix)
            ]
        _cqfunc.__doc__ = "Columns that start with {}".format(self._prefix)
        kwargs['func'] = _cqfunc
        super().__init__(**kwargs)

    def __repr__(self):
        return "<ColumnQualifier: Columns starting with {}>".format(
            self._prefix)


class OfDtypes(ColumnQualifier):
    """Selectes all columns that are of a given dtypes.

    Use `dtypes=np.number` to qualify all numeric columns.

    Parameters
    ----------
    dtypes : object or list of objects
        The dtype or dtypes which qualify columns. Support all valid arguments
        to the `include` parameter of pandas.DataFrame.select_dtypes().
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        ColumnQualifier. See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import numpy as np;
        >>> df = pd.DataFrame(
        ...    [[8.2,'a',5],[5.1,'b',7]], [1,2], ['ph', 'grade', 'age'])
        >>> cq = pdp.cq.OfDtypes(np.number)
        >>> cq(df)
        ['ph', 'age']
        >>> cq = pdp.cq.OfDtypes([np.number, object])
        >>> cq(df)
        ['ph', 'grade', 'age']
        >>> cq = pdp.cq.OfDtypes(np.int64)
        >>> cq
        <ColumnQualifier: With dtypes in <class 'numpy.int64'>>
        >>> cq(df)
        ['age']
    """

    def __init__(self, dtypes, **kwargs):
        self._dtypes = dtypes
        self._dtypes_str = _list_str(self._dtypes)
        def _cqfunc(df):  # noqa: E306
            return list(df.select_dtypes(include=self._dtypes).columns)
        _cqfunc.__doc__ = "Columns of dtypes {}".format(self._dtypes_str)
        kwargs['func'] = _cqfunc
        super().__init__(**kwargs)

    def __repr__(self):
        return "<ColumnQualifier: With dtypes in {}>".format(self._dtypes_str)


class WithAtMostMissingValues(ColumnQualifier):
    """Selectes all columns with no more than X missing values.

    Parameters
    ----------
    n_missing : int
        The maximum number of missing values with which columns can still
        qualify.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        ColumnQualifier. See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import numpy as np;
        >>> df = pd.DataFrame(
        ...    [[None, 1, 2],[None, None, 5]], [1,2], ['ph', 'grade', 'age'])
        >>> cq = pdp.cq.WithAtMostMissingValues(1)
        >>> cq
        <ColumnQualifier: With at most 1 missing values>
        >>> cq(df)
        ['grade', 'age']
    """

    def __init__(self, n_missing, **kwargs):
        self._n_missing = n_missing
        def _cqfunc(df):  # noqa: E306
            return list(df.columns[df.isna().sum() <= self._n_missing])
        _cqfunc.__doc__ = "Columns with at most {} missing values".format(
            self._n_missing)
        kwargs['func'] = _cqfunc
        super().__init__(**kwargs)

    def __repr__(self):
        return "<ColumnQualifier: With at most {} missing values>".format(
            self._n_missing)


class WithoutMissingValues(WithAtMostMissingValues):
    """Selectes all columns with no missing values.

    Parameters
    ----------
    **kwargs
        Accepts all keyword arguments of the constructor of ColumnQualifier.
        See the documentation of ColumnQualifier for details.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import numpy as np;
        >>> df = pd.DataFrame(
        ...    [[None, 1, 2],[None, None, 5]], [1,2], ['ph', 'grade', 'age'])
        >>> cq = pdp.cq.WithoutMissingValues()
        >>> cq
        <ColumnQualifier: Without missing values>
        >>> cq(df)
        ['age']
    """

    def __init__(self, **kwargs):
        kwargs['n_missing'] = 0
        super().__init__(**kwargs)

    def __repr__(self):
        return "<ColumnQualifier: Without missing values>"
