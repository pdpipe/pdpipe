"""Fittable conditions for pdpipe."""

import pandas

from .shared import _list_str


class UnfittedConditionError(Exception):
    """An exception raised when a (non-fit) transform is attempted with an
    unfitted condition.
    """


class Condition(object):
    """A fittable condition that returns a boolean value from a dataframe.

    Parameters
    ----------
    func : callable
        A callable that given an input pandas.DataFrame objects returns a
        boolean value.
    fittable : bool, default False
        If set to True, this condition becomes fittable, and `func` is not
        called on calls of `transform()` of a fitted object. If set to False,
        the default, `func` is called on every call to transform. False by
        default.
    error_message : str, default None
        A string that describes the error when the condition fails.

    Example
    -------
    >>> import numpy as np; import pdpipe as pdp;
    >>> cond = pdp.cond.Condition(lambda df: 'a' in df.columns)
    >>> cond
    <pdpipe.Condition: By function>
    >>> col_drop = pdp.ColDrop(['lbl'], prec=cond)
    """

    def __init__(self, func, fittable=None, error_message=None):
        self._func = func
        self._fittable = fittable
        if error_message is not None:
            self.error_message = error_message

    def __call__(self, df):
        """Returns column labels of qualified columns from an input dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe on which the condition is checked.

        Returns
        -------
        bool
            Either True of False.
        """
        try:
            return self.transform(df)
        except UnfittedConditionError:
            return self.fit_transform(df)

    def fit_transform(self, df):
        """Fits this condition and returns the result.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe on which the condition is checked.

        Returns
        -------
        bool
            Either True or False.
        """
        self._result = self._func(df)
        return self._result

    def fit(self, df):
        """Fits this condition on the input dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe on which the condition is checked.
        """
        self.fit_transform(df)

    def transform(self, df):
        """Returns the result of this condition.

        Is this Condition is fittable, it will return the result that was
        determined when fitted, if it's fitted, and throw an exception
        if it is not.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe on which the condition is checked.

        Returns
        -------
        bool
            Either True or False.
        """
        if not self._fittable:
            return self._func(df)
        try:
            return self._result
        except AttributeError:
            raise UnfittedConditionError

    def __repr__(self):
        fstr = ''
        if self._func.__doc__:  # pragma: no cover
            fstr = f' - {self._func.__doc__}'
        return f"<pdpipe.Condition: By function{fstr}>"

    # --- overriding boolean operators ---

    # need this because inner-scope functions aren't pickle-able
    class _AndCondition(object):

        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __call__(self, df):
            return self.first(df) and self.second(df)

    def __and__(self, other):
        try:
            _func = Condition._AndCondition(self._func, other._func)
            _func.__doc__ = (
                f"{self._func.__doc__ or 'Anonymous condition 1'} AND "
                f"{other._func.__doc__ or 'Anonymous condition 2'}"
            )
            return Condition(func=_func)
        except AttributeError:
            return NotImplemented

    class _XorCondition(object):

        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __call__(self, df):
            return self.first(df) != self.second(df)

    def __xor__(self, other):
        try:
            _func = Condition._XorCondition(self._func, other._func)
            _func.__doc__ = (
                f"{self._func.__doc__ or 'Anonymous condition 1'} XOR "
                f"{other._func.__doc__ or 'Anonymous condition 2'}"
            )
            return Condition(func=_func)
        except AttributeError:
            return NotImplemented

    class _OrCondition(object):

        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __call__(self, df):
            return self.first(df) or self.second(df)

    def __or__(self, other):
        try:
            _func = Condition._OrCondition(self._func, other._func)
            _func.__doc__ = (
                f"{self._func.__doc__ or 'Anonymous condition 1'} OR "
                f"{other._func.__doc__ or 'Anonymous condition 2'}"
            )
            return Condition(func=_func)
        except AttributeError:
            return NotImplemented

    class _NotCondition(object):

        def __init__(self, first):
            self.first = first

        def __call__(self, df):
            return not self.first(df)

    def __invert__(self):
        _func = Condition._NotCondition(self._func)
        _func.__doc__ = f"NOT {self._func.__doc__ or 'Anonymous condition'}"
        return Condition(func=_func)


class PerColumnCondition(Condition):
    """Checks whether the columns of input dataframes satisfy a condition set.

    Parameters
    ----------
    conditions : callable or list-like
        The condition, or set of conditions, that columns of input dataframes
        must satisfy. Conditions are callables that accept a `pandas.Series`
        object and return a `bool` value.
    conditions_reduce : str, default 'all'
        How condition satisfaction results are reduced per-column, in case of
        multiple conditions. 'all' requires a column to satisfy all conditions,
        while 'any' requires at least one condition to be satisfied.
    columns_reduce : str, default 'all'
        How condition satisfaction results are reduced among multiple columns.
        'all' requires all columns of input dataframes to satisfy the given
        condition (in the case of multiple conditions, behaviour is determined
        by the `condition_reduce` parameter), while 'any' requires at least one
        column to satisfy it.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        Condition. See the documentation of Condition for details.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp; import numpy as np;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.PerColumnCondition(
    ...     conditions=lambda x: x.dtype == np.int64,
    ... )
    >>> cond
    <pdpipe.Condition: Dataframes with all columns satisfying all \
conditions: anonymous condition>
    >>> cond(df)
    False
    >>> cond = pdp.cond.PerColumnCondition(
    ...     conditions=lambda x: x.dtype == np.int64,
    ...     columns_reduce='any',
    ... )
    >>> cond(df)
    True
    >>> cond = pdp.cond.PerColumnCondition(
    ...     conditions=[
    ...         lambda x: x.dtype == np.int64,
    ...         lambda x: x.dtype == object,
    ...     ],
    ... )
    >>> cond(df)
    False
    >>> cond = pdp.cond.PerColumnCondition(
    ...     conditions=[
    ...         lambda x: x.dtype == np.int64,
    ...         lambda x: x.dtype == object,
    ...     ],
    ...     conditions_reduce='any',
    ... )
    >>> cond(df)
    True
    """

    class _ConditionFunction(object):

        def __init__(self, conditions, cond_reduce, col_reduce):
            self.conditions = conditions
            self.cond_reduce = cond_reduce
            self.col_reduce = col_reduce

        def __call__(self, df):
            return self.col_reduce([
                self.cond_reduce([
                    cond(df[lbl])
                    for cond in self.conditions
                ])
                for lbl in df.columns
            ])

    def __init__(self, conditions, conditions_reduce=None, columns_reduce=None,
                 **kwargs):
        # handling default args and input types
        if not hasattr(conditions, '__iter__'):
            conditions = [conditions]
        if conditions_reduce is None:
            conditions_reduce = 'all'
        if columns_reduce is None:
            columns_reduce = 'all'
        # building class attributes
        self._conditions = conditions
        self._cond_reduce_str = conditions_reduce
        self._col_reduce_str = columns_reduce
        self._conditions_str = ', '.join([
            c.__doc__ or 'anonymous condition'
            for c in conditions
        ])
        if conditions_reduce == 'all':
            self._cond_reduce = all
        elif conditions_reduce == 'any':
            self._cond_reduce = any
        else:
            raise ValueError((
                "The only valid arguments to the `conditions_reduce` parameter"
                " of PerColumnCondition are 'all' and 'any'!"
            ))
        if columns_reduce == 'all':
            self._col_reduce = all
        elif columns_reduce == 'any':
            self._col_reduce = any
        else:
            raise ValueError((
                "The only valid arguments to the `columns_reduce` parameter"
                " of PerColumnCondition are 'all' and 'any'!"
            ))
        # building resulting function
        _func = PerColumnCondition._ConditionFunction(
            conditions=self._conditions,
            cond_reduce=self._cond_reduce,
            col_reduce=self._col_reduce,
        )
        doc_str = "Dataframes with {} columns satisfying {} conditions: {}"
        self._func_doc = doc_str.format(
            self._col_reduce_str, self._cond_reduce_str, self._conditions_str)
        _func.__doc__ = self._func_doc
        kwargs['func'] = _func
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<pdpipe.Condition: {self._func_doc}>"


class HasAllColumns(Condition):
    """Checks whether input dataframes contain a list of columns.

    Parameters
    ----------
    labels : single label or list-like
        Column labels to check for.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        Condition. See the documentation of Condition for details.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.HasAllColumns('num')
    >>> cond
    <pdpipe.Condition: Has all columns in num>
    >>> cond(df)
    True
    >>> cond = pdp.cond.HasAllColumns(['num', 'chr'])
    >>> cond(df)
    True
    >>> cond = pdp.cond.HasAllColumns(['num', 'gar'])
    >>> cond(df)
    False
    """

    def __init__(self, labels, **kwargs):
        if isinstance(labels, str) or not hasattr(labels, '__iter__'):
            labels = [labels]
        self._labels = labels
        self._labels_str = _list_str(self._labels)
        def _func(df):  # noqa: E306
            return all([
                lbl in df.columns
                for lbl in self._labels
            ])
        _func.__doc__ = f"Dataframes with columns {self._labels_str}"
        super_kwargs = {
            "error_message": (
                f"Not all required columns {self._labels_str}"
                " present in the input dataframe."
            )
        }
        super_kwargs.update(**kwargs)
        super_kwargs['func'] = _func
        super().__init__(**super_kwargs)

    def __repr__(self):
        return f"<pdpipe.Condition: Has all columns in {self._labels_str}>"


class ColumnsFromList(PerColumnCondition):
    """Checks whether input dataframes contain columns from a list.

    Parameters
    ----------
    labels : single label or list-like
        Column labels to check for.
    columns_reduce : str, default 'all'
        How condition satisfaction results are reduced among multiple columns.
        'all' requires all columns of input dataframes to satisfy the given
        condition, while 'any' requires at least one column to satisfy it.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        Condition. See the documentation of Condition for details.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.ColumnsFromList('num')
    >>> cond
    <pdpipe.Condition: Dataframes with all columns satisfying all \
conditions: Series with labels in num>
    >>> cond(df)
    False
    >>> cond = pdp.cond.ColumnsFromList(['num', 'chr', 'nur'])
    >>> cond(df)
    True
    >>> cond = pdp.cond.ColumnsFromList(
    ...     ['num', 'gar'], columns_reduce='any')
    >>> cond(df)
    True
    """

    class _SeriesLblCondition(object):

        def __init__(self, labels):
            self.labels = labels

        def __call__(self, series):
            return series.name in self.labels

    def __init__(self, labels, columns_reduce=None, **kwargs):
        if isinstance(labels, str) or not hasattr(labels, '__iter__'):
            labels = [labels]
        self._labels = labels
        self._labels_str = _list_str(self._labels)
        _func = ColumnsFromList._SeriesLblCondition(self._labels)
        _func.__doc__ = f"Series with labels in {self._labels_str}"
        kwargs['conditions'] = [_func]
        kwargs['columns_reduce'] = columns_reduce
        super().__init__(**kwargs)


class HasNoColumn(Condition):
    """Checks whether input dataframes contains no column from a list.

    Parameters
    ----------
    labels : single label or list-like
        Column labels to check for.
    **kwargs
        Additionaly accepts all keyword arguments of the constructor of
        Condition. See the documentation of Condition for details.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.HasNoColumn('num')
    >>> cond
    <pdpipe.Condition: Has no column in num>
    >>> cond(df)
    False
    >>> cond = pdp.cond.HasNoColumn(['num', 'gar'])
    >>> cond(df)
    False
    >>> cond = pdp.cond.HasNoColumn(['ph', 'gar'])
    >>> cond(df)
    True
    """

    class _NoColumnsFunc(object):

        def __init__(self, labels):
            self.labels = labels

        def __call__(self, df):
            return all([
                lbl not in df.columns
                for lbl in self.labels
            ])

    def __init__(self, labels, **kwargs):
        if isinstance(labels, str) or not hasattr(labels, '__iter__'):
            labels = [labels]
        self._labels = labels
        self._labels_str = _list_str(self._labels)
        _func = HasNoColumn._NoColumnsFunc(self._labels)
        _func.__doc__ = f"Dataframes with no column from {self._labels_str}"
        super_kwargs = {
            "error_message": (
                f"One or more of the prohibited columns {self._labels_str}"
                " present in the input dataframe."
            )
        }
        super_kwargs.update(**kwargs)
        super_kwargs['func'] = _func
        super().__init__(**super_kwargs)

    def __repr__(self):
        return f"<pdpipe.Condition: Has no column in {self._labels_str}>"


class HasAtMostMissingValues(Condition):
    """Checks whether input dataframes has no more than X missing values
    across all columns.

    Parameters
    ----------
    n_missing : int or float
        If int, then interpreted as the maximal allowed number of missing
        values in input dataframes. If float, interpreted as the maximal
        allowed ratio of missing values in input dataframes.
    **kwargs
        Additionally accepts all keyword arguments of the constructor of
        Condition. See the documentation of Condition for details.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[None,'a',5],[5,None,7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.HasAtMostMissingValues(1)
    >>> cond
    <pdpipe.Condition: Has at most 1 missing values>
    >>> cond(df)
    False
    >>> cond = pdp.cond.HasAtMostMissingValues(2)
    >>> cond(df)
    True
    >>> cond = pdp.cond.HasAtMostMissingValues(0.4)
    >>> cond(df)
    True
    >>> cond = pdp.cond.HasAtMostMissingValues(0.2)
    >>> cond(df)
    False
    """

    class _IntMissingValuesFunc(object):

        def __init__(self, n_missing):
            self.n_missing = n_missing

        def __call__(self, df):
            nmiss = df.isna().sum().sum()
            return nmiss <= self.n_missing

    class _FloatMissingValuesFunc(object):

        def __init__(self, n_missing):
            self.n_missing = n_missing

        def __call__(self, df):
            nmiss = df.isna().sum().sum()
            return (nmiss / df.size) <= self.n_missing

    def __init__(self, n_missing, **kwargs):
        self._n_missing = n_missing
        if isinstance(n_missing, int):
            _func = HasAtMostMissingValues._IntMissingValuesFunc(n_missing)
        elif isinstance(n_missing, float):
            _func = HasAtMostMissingValues._FloatMissingValuesFunc(n_missing)
        else:
            raise ValueError("n_missing should be of type int or float!")
        _func.__doc__ = (
            f"Dataframes with at most {self._n_missing} missing values"
        )
        super_kwargs = {
            "error_message": (
                "Input dataframe cannot have more than"
                f" {self._n_missing} missing values."
            )
        }
        super_kwargs.update(**kwargs)
        super_kwargs['func'] = _func
        super().__init__(**super_kwargs)

    def __repr__(self):
        return f"<pdpipe.Condition: " \
               f"Has at most {self._n_missing} missing values>"


class HasNoMissingValues(HasAtMostMissingValues):
    """Checks whether input dataframes has no missing values.

    Parameters
    ----------
    **kwargs
        Accepts all keyword arguments of the constructor of Condition. See the
        documentation of Condition for details.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[None,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.HasNoMissingValues()
    >>> cond
    <pdpipe.Condition: Has no missing values>
    >>> cond(df)
    False
    """

    def __init__(self, **kwargs):
        super_kwargs = {
            "error_message": "Input dataframe cannot contain missing values."
        }
        super_kwargs.update(**kwargs)
        super_kwargs['n_missing'] = 0
        super().__init__(**super_kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: Has no missing values>"


def _AlwaysTrue(df: pandas.DataFrame) -> bool:
    """A function that always returns True."""
    return True


class AlwaysTrue(Condition):
    """A condition letting all dataframes through, always returning True.

    Example
    -------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = pdp.cond.AlwaysTrue()
    >>> cond
    <pdpipe.Condition: AlwaysTrue>
    >>> cond(df)
    True
    """

    def __init__(self, **kwargs):
        super_kwargs = {}
        super_kwargs.update(**kwargs)
        super_kwargs['func'] = _AlwaysTrue
        super().__init__(**super_kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: AlwaysTrue>"
