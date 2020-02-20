"""Fittable conditions for pdpipe.

In `pdipe`, pipeline stages have two optional constructor parameters that
accept callables that are treated as conditions: `prec` and `skip`. Both assume
input callables can accept a pandas.Dataframe object as input and return either
True or False. `prec` - representing the stage's precondition - determines
whether a stage *can* be applied to an input dataframe, while `skip` -
representing the stage's skip condition - determines whether it *should* be
applied. Accordingly, a stage throws a `FailedPreconditionError` if its
precondition is not statisfied, while it is skipped if its skip-condition is
not statisfied.

This module - `pdpipe.cond` - provides a way to easily generate `Condition`
objects, which are callable, and can easily be made fittable - to have their
result determined in fit time and preserved for future transforms. This
enables the creation pipeline stages that their effective inclusion in the
pipeline is determined when `fit_transform` is called; for example, whether
dimensionality reduction is required - once this decision is done in training
time it should be maintained for all future transforms of data (in test and
validation sets or in production).
"""

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
    fittable : bool, default True
        If set to false, this condition becomes unfittable, and `func` is
        called on every call to transform. True by default.

    Example
    -------
        >>> import numpy as np; import pdpipe as pdp;
        >>> cond = pdp.cond.Condition(lambda df: 'a' in df.columns)
        >>> cond
        <pdpipe.Condition: By function>
        >>> col_drop = pdp.ColDrop(['lbl'], prec=cond)
    """

    def __init__(self, func, fittable=None):
        if fittable is None:
            fittable = True
        self._func = func
        self._fittable = fittable

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
            Either True of False.
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
            Either True of False.
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
            fstr = ' - {}'.format(self._cqfunc.__doc__)
        return "<pdpipe.Condition: By function{}>".format(fstr)

    # --- overriding boolean operators ---

    def __and__(self, other):
        try:
            ofunc = other._func
            def _func(df):  # noqa: E306
                return self._func(df) and ofunc(df)
            _func.__doc__ = '{} AND {}'.format(
                self._func.__doc__ or 'Anonymous condition 1',
                other._func.__doc__ or 'Anonymous condition 2',
            )
            return Condition(func=_func)
        except AttributeError:
            return NotImplemented

    def __xor__(self, other):
        try:
            ofunc = other._func
            def _func(df):  # noqa: E306
                return self._func(df) != ofunc(df)
            _func.__doc__ = '{} XOR {}'.format(
                self._func.__doc__ or 'Anonymous condition 1',
                other._func.__doc__ or 'Anonymous condition 2',
            )
            return Condition(func=_func)
        except AttributeError:
            return NotImplemented

    def __or__(self, other):
        try:
            ofunc = other._func
            def _func(df):  # noqa: E306
                return self._func(df) or ofunc(df)
            _func.__doc__ = '{} OR {}'.format(
                self._func.__doc__ or 'Anonymous condition 1',
                other._func.__doc__ or 'Anonymous condition 2',
            )
            return Condition(func=_func)
        except AttributeError:
            return NotImplemented

    def __invert__(self):
        def _func(df):  # noqa: E306
            return not self._func(df)
        _func.__doc__ = 'NOT {}'.format(
            self._func.__doc__ or 'Anonymous condition'
        )
        return Condition(func=_func)


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
        _func.__doc__ = "Dataframes with colums {}".format(
            self._labels_str)
        kwargs['func'] = _func
        super().__init__(**kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: Has all columns in {}>".format(
            self._labels_str)


class HasAnyColumn(Condition):
    """Checks whether input dataframes contain at least one column from a list.

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
        >>> cond = pdp.cond.HasAnyColumn('num')
        >>> cond
        <pdpipe.Condition: Has any column in num>
        >>> cond(df)
        True
        >>> cond = pdp.cond.HasAnyColumn(['num', 'chr'])
        >>> cond(df)
        True
        >>> cond = pdp.cond.HasAnyColumn(['num', 'gar'])
        >>> cond(df)
        True
    """

    def __init__(self, labels, **kwargs):
        if isinstance(labels, str) or not hasattr(labels, '__iter__'):
            labels = [labels]
        self._labels = labels
        self._labels_str = _list_str(self._labels)
        def _func(df):  # noqa: E306
            return any([
                lbl in df.columns
                for lbl in self._labels
            ])
        _func.__doc__ = "Dataframes with any colum from {}".format(
            self._labels_str)
        kwargs['func'] = _func
        super().__init__(**kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: Has any column in {}>".format(
            self._labels_str)


class HasNoColumn(Condition):
    """Checks whether input dataframes contain at no column from a list.

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

    def __init__(self, labels, **kwargs):
        if isinstance(labels, str) or not hasattr(labels, '__iter__'):
            labels = [labels]
        self._labels = labels
        self._labels_str = _list_str(self._labels)
        def _func(df):  # noqa: E306
            return all([
                lbl not in df.columns
                for lbl in self._labels
            ])
        _func.__doc__ = "Dataframes with no colum from {}".format(
            self._labels_str)
        kwargs['func'] = _func
        super().__init__(**kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: Has no column in {}>".format(
            self._labels_str)


class HasAtMostMissingValues(Condition):
    """Checks whether input dataframes has no more than X missing values.

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

    def __init__(self, n_missing, **kwargs):
        self._n_missing = n_missing
        if isinstance(n_missing, int):
            def _func(df):
                nmiss = df.isna().sum().sum()
                return nmiss <= self._n_missing
        elif isinstance(n_missing, float):
            def _func(df):
                nmiss = df.isna().sum().sum()
                return nmiss / df.size <= self._n_missing
        _func.__doc__ = "Dataframes with at most {} missing values".format(
            self._n_missing)
        kwargs['func'] = _func
        super().__init__(**kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: Has at most {} missing values>".format(
            self._n_missing)


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
        kwargs['n_missing'] = 0
        super().__init__(**kwargs)

    def __repr__(self):
        return "<pdpipe.Condition: Has no missing values>"
