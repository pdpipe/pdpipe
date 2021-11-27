"""Defines pipelines for processing pandas.DataFrame-based datasets.

>>> import pdpipe as pdp
>>> pipeline = pdp.ColDrop('Name') + pdp.Bin({'Speed': [0,5]})
>>> pipeline = pdp.ColDrop('Name').Bin({'Speed': [0,5]}, drop=True)

## Creating pipeline stages that operate on column subsets

Many pipeline stages in pdpipe operate on a subset of columns, allowing the
caller to determine this subset by either providing a fixed set of column
labels or by providing a callable that determines the column subset dynamically
from input dataframes. The `pdpipe.cq` module addresses a unique but important
use case of fittable column qualifier, which is to dynamically extract a column
subset on stage fit time, but keep it fixed for future transformations.

As a general rule, every pipeline stage in pdpipe that supports the `columns`
parameter should inherently support fittable column qualifier, and generally
the correct interpretation of both single and multiple labels as arguments. To
unify the implementation of such functionality, and ease of creation of new
pipeline stages, such columns should be created by extending the
ColumnsBasedPipelineStage base class, found in this module (`pdpipe.core`).

The main interface of sub-classes of this base class with it is through the
`columns`, `exclude_columns` and `none_columns` constructor arguments, and the
"private" `_get_columns(df, fit)` method:

* Any extending subclass should accept the `columns` constructor parameter
  and forward it, without transforming it, to the constructor of
  ColumnsBasedPipelineStage. E.g.
  `super().__init__(columns=columns, **kwargs)`. See the implementation of
  any such extending class for a more complete example.

* Extending subclasses can decide if they want to expose the
  `exclude_columns` parameter or not. Note that most of its functionality
  can anyway be gained by providing the `columns` parameter with a column
  qualifier object that is a difference between two column qualifiers; e.g.
  `columns=cq.OfDtype(np.number) - cq.OfDtype(np.int64)` is equivalent to
  providing `columns=cq.OfDtype(np.number),
  exclude_columns=cq.OfDtype(np.int64)`. However, exposing the
  `exclude_columns` parameter can allow for specific unique behaviours; for
  example, if the `none_columns` parameter - which configures the behavior
  when `columns` is provided with `None` - is set with
  a `cq.OfDtypes('category')` column qualifier, which means that all
  categorical columns are selected when `columns=None`, then exposing
  `exclude_columns` allows for easy specification of the "all categorical
  columns except X" by just giving a column qualifier capturing X to
  `exclude_columns`, instead of having to reconstruct the default column
  qualifier by hand and substract from it the one representing X.

* When wishing to get the subset of columns to operate on, in
  `fit_transform` or `transform` time, it is attained by calling
  `self._get_columns(df, fit=True)` (or with `fit=False` if just
  transforming), providing it the input dataframe.

* Additionally, to get a description and application message with a nice
  string representation of the list of columns to operate on, the
  `desc_temp` constructor parameter of ColumnsBasedPipelineStage can be
  provided with a format string with a place holder where the column list
  should go. E.g. `"Drop columns {}"` for the DropCol pipeline stage.

There are two correct ways to extend it, depending on whether the pipeline
stage you're creating is inherently fittable or not:

1. If the stage is NOT inherently fittable, then the ability to accept
   fittable column qualifier objects makes it so. However, to enable
   extending subclasses to implement their transformation using a single
   method, they can simply implement the abstract method
   `_transformation(self, df, verbose, fit)`. It should treat the `df` and
   `verbose` parameters normally, but forward the `fit` parameter to the
   `_get_columns` method when calling it. This is enough to get a pipeline
   stage with the desired behavior, with the super-class handling all the
   fit/transform functionality.

2. If the stage IS inherently fittable, then do not use the
   `_transformation` abstract method (it has to be implemented, so just
   have it raise a NotImplementedError). Instead, simply override the
   `_fit_transform` and `_transform` method of ColumnsBasedPipelineStage,
   calling the `fit` parameter of the `_get_columns` method with the
   correct arguement: `True` when fit-transforming and `False` when
   transforming.

Again, taking a look at the VERY concise implementation of simple columns-based
stages, like ColDrop or ValDrop in `pdpipe.basic_stages`, will probably make
things clearer, and you can use those implementations as a template for yours.
"""

import sys
import abc
import time
import inspect
import collections
import textwrap

try:
    from pympler.asizeof import asizeof
except ImportError:
    from sys import getsizeof as asizeof

from .cq import is_fittable_column_qualifier, AllColumns
from .shared import _get_args_list
from .exceptions import (
    FailedPreconditionError,
    FailedPostconditionError,
    UnfittedPipelineStageError,
    PipelineApplicationError
)


# === loading stage attributes ===

def __get_append_stage_attr_doc(class_obj):
    doc = class_obj.__doc__
    first_line = doc[0:doc.find('.') + 1]
    if "An" in first_line:
        new_first_line = first_line.replace("An", "Creates and adds an", 1)
    else:
        new_first_line = first_line.replace("A", "Creates and adds a", 1)
    new_first_line = new_first_line[0:-1] + (
        " to this pipeline stage.")
    return doc.replace(first_line, new_first_line, 1)


def __load_stage_attribute__(class_obj):

    def _append_stage_func(self, *args, **kwds):
        # self is always a PdPipelineStage
        return self + class_obj(*args, **kwds)
    _append_stage_func.__doc__ = __get_append_stage_attr_doc(class_obj)
    _append_stage_func.__name__ = class_obj.__name__  # .lower()
    _append_stage_func.__signature__ = inspect.signature(class_obj.__init__)
    setattr(PdPipelineStage, class_obj.__name__, _append_stage_func)

    # unbound_method = types.MethodType(_append_stage_func, class_obj)
    # setattr(class_obj, class_obj.__name__, unbound_method)


def __load_stage_attributes_from_module__(module_name):
    module_obj = sys.modules[module_name]
    for name, obj in inspect.getmembers(module_obj):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            class_obj = getattr(module_obj, name)
            if issubclass(class_obj, PdPipelineStage) and (
                    class_obj.__name__ != 'PdPipelineStage'):
                __load_stage_attribute__(class_obj)


# === basic classes ===

class PdpApplicationContext(dict):
    """An object encapsulating the application context of a pipeline.

    It is meant to communicate data, information and variables between
    different stages of a pipeline.

    Parameters
    ----------
    fit_context : PdpApplicationContext, optional
        Another application context object, representing the application
        context of a previous fit of the pipelline this application context
        is initialized for. Optional.
    """

    def __init__(self, fit_context=None):
        self.__locked__ = False
        self._fit_context__ = fit_context

    def __setitem__(self, key, value):
        if not self.__locked__:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        if not self.__locked__:
            super().__delitem__(key)

    def pop(self, key, default):
        """If key is in the dictionary, remove it and return its value, else
        return default. If default is not given and key is not in the
        dictionary, a KeyError is raised.
        """
        if not self.__locked__:
            return super().pop(key, default)
        return super().__getitem__(key)

    def clear(self):
        """Remove all items from the dictionary."""
        if not self.__locked__:
            super().clear()

    def popitem(self):
        """Not implemented!"""
        raise NotImplementedError

    def update(self, other):
        """Update the dictionary with the key/value pairs from other,
        overwriting existing keys. Return None.
        update() accepts either another dictionary object or an iterable of
        key/value pairs (as tuples or other iterables of length two). If
        keyword arguments are specified, the dictionary is then updated with
        those key/value pairs: d.update(red=1, blue=2).
        """
        if not self.__locked__:
            super().update(other)

    def lock(self):
        """Locks this application context for changes."""
        self.__locked__ = True

    def fit_context(self):
        """Returns a locked PdpApplicationContext object of a previous fit."""
        return self._fit_context__


class PdPipelineStage(abc.ABC):
    """A stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Additionally, if true, a
        pdpipe.FailedPostconditionError is raised if an expected post-condition
        does not hold for an output dataframe (after pipeline application).
        Otherwise pipeline application continues uninterrupted.
    exmsg : str, default None
        The message of the exception that is raised on a failed
        precondition if exraise is set to True. A default message is used
        if None is given.
    desc : str, default None
        A short description of this stage, used as its string representation.
        A default description is used if None is given.
    prec : callable, default None
        This can be assigned a callable that returns boolean values for input
        dataframes, which will be used to determine whether input dataframes
        satisfy the preconditions for this pipeline stage (see the `exraise`
        parameter for the behaviour of failed preconditions). See `pdpipe.cond`
        for more information on specialised Condition objects.
    post : callable, default None
        This can be assigned a callable that returns boolean values for input
        dataframes, which will be used to determine whether input dataframes
        satisfy the postconditions for this pipeline stage (see the `exraise`
        parameter for the behaviour of failed postconditions). See
        `pdpipe.cond` for more information on specialised Condition objects.
    skip : callable, default None
        This can be assigned a callable that returns boolean values for input
        dataframes, which will be used to determine whether this stage should
        be skipped for input dataframes - if the callable returns True for an
        input dataframe, this stage will be skipped. See `pdpipe.cond` for more
        information on specialised Condition objects.
    name : str, default ''
        The name of this stage. Pipelines can be sliced by this name.

    Attributes
    ----------
    fit_context : `PdpApplicationContext`
        An application context object that is only re-initialized before
        `fit_transform` calls, and is locked after pipeline application. It is
        injected into the PipelineStage by the encapsulating pipeline object.
    application_context : `PdpApplicationContext`
        An application context object that is re-initialized before every
        pipeline application (so, also during transform operations of fitted
        pipelines), and is locked after pipeline application.It is injected
        into the PipelineStage by the encapsulating pipeline object.
    """

    _DEF_EXC_MSG = 'Precondition failed in stage {}!'
    _DEF_DESCRIPTION = 'A pipeline stage.'
    _INIT_KWARGS = ['exraise', 'exmsg', 'desc', 'prec', 'skip', 'name']

    def __init__(self, exraise=True, exmsg=None, desc=None, prec=None,
                 post=None, skip=None, name=''):
        if not isinstance(name, str):
            raise ValueError(
                f"'name' must be a str, not {type(name).__name__}."
            )
        if desc is None:
            desc = PdPipelineStage._DEF_DESCRIPTION
        if exmsg is None:
            exmsg = PdPipelineStage._DEF_EXC_MSG.format(desc)

        self._exraise = exraise
        self._exmsg = exmsg
        self._exmsg_post = exmsg.replace(
            'precondition', 'postcondition').replace(
            'Precondition', 'Postcondition')
        self._desc = desc
        self._prec_arg = prec
        self._post_arg = post
        self._skip = skip
        self._appmsg = f"{name + ': ' if name else ''}{desc}"
        self._name = name
        self.fit_context: PdpApplicationContext = None
        self.application_context: PdpApplicationContext = None
        self.is_fitted = False

    @classmethod
    def _init_kwargs(cls):
        return cls._INIT_KWARGS

    @abc.abstractmethod
    def _prec(self, df):  # pylint: disable=R0201,W0613
        """Returns True if this stage can be applied to the given dataframe."""
        raise NotImplementedError

    def _compound_prec(self, df):
        if self._prec_arg:
            return self._prec_arg(df)
        return self._prec(df)

    def _post(self, df):  # pylint: disable=R0201,W0613
        """Returns True if this stage resulted in an expected output frame."""
        return True

    def _compound_post(self, df):
        if self._post_arg:
            return self._post_arg(df)
        return self._post(df)

    def _fit_transform(self, df, verbose):
        """Fits this stage and transforms the input dataframe."""
        return self._transform(df, verbose)

    def _is_fittable(self):
        if self.__class__._fit_transform == PdPipelineStage._fit_transform:
            return False
        return True

    def _raise_precondition_error(self):
        try:
            raise FailedPreconditionError(
                f"{self._exmsg} [Reason] {self._prec_arg.error_message}")
        except AttributeError:
            raise FailedPreconditionError(self._exmsg)

    def _raise_postcondition_error(self):
        try:
            raise FailedPostconditionError(
                f"{self._exmsg_post} [Reason] {self._post_arg.error_message}")
        except AttributeError:
            raise FailedPostconditionError(self._exmsg_post)

    @abc.abstractmethod
    def _transform(self, df, verbose):
        """Transforms the given dataframe without fitting this stage."""
        raise NotImplementedError("_transform method not implemented!")

    def apply(self, df, exraise=None, verbose=False):
        """Applies this pipeline stage to the given dataframe.

        If the stage is not fitted fit_transform is called. Otherwise,
        transform is called.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to which this pipeline stage will be applied.
        exraise : bool, default None
            Override preconditions and postconditions behaviour for this call.
            If None, the default behaviour of this stage is used, as determined
            by the exraise constructor parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            is checked but before the application of the pipeline stage.
            Defaults to False.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if exraise is None:
            exraise = self._exraise
        if self._skip and self._skip(df):
            return df
        if self._compound_prec(df=df):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            if self.is_fitted:
                res_df = self._transform(df, verbose=verbose)
            else:
                res_df = self._fit_transform(df, verbose=verbose)
            if exraise and not self._compound_post(df=res_df):
                self._raise_postcondition_error()
            return res_df
        if exraise:
            self._raise_precondition_error()
        return df

    __call__ = apply

    def fit_transform(self, X, y=None, exraise=None, verbose=False):
        """Fits this stage and transforms the given dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to transform and fit this pipeline stage by.
        y : array-like, optional
            Targets for supervised learning.
        exraise : bool, default None
            Override preconditions and postconditions behaviour for this call.
            If None, the default behaviour of this stage is used, as determined
            by the exraise constructor parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            is checked but before the application of the pipeline stage.
            Defaults to False.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if exraise is None:
            exraise = self._exraise
        if self._compound_prec(X):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            res_df = self._fit_transform(X, verbose=verbose)
            if exraise and not self._compound_post(df=res_df):
                self._raise_postcondition_error()
            return res_df
        if exraise:
            self._raise_precondition_error()
        return X

    def fit(self, X, y=None, exraise=None, verbose=False):
        """Fits this stage without transforming the given dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to be transformed.
        y : array-like, optional
            Targets for supervised learning.
        exraise : bool, default None
            Override preconditions and postconditions behaviour for this call.
            If None, the default behaviour of this stage is used, as determined
            by the exraise constructor parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            is checked but before the application of the pipeline stage.
            Defaults to False.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if exraise is None:
            exraise = self._exraise
        if self._compound_prec(X):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            res_df = self._fit_transform(X, verbose=verbose)
            if exraise and not self._compound_post(df=res_df):
                self._raise_postcondition_error()
            return X
        if exraise:
            self._raise_precondition_error()
        return X

    def transform(self, X, y=None, exraise=None, verbose=False):
        """Transforms the given dataframe without fitting this stage.

        If this stage is fittable but is not fitter, an
        UnfittedPipelineStageError is raised.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to be transformed.
        y : array-like, optional
            Targets for supervised learning.
        exraise : bool, default None
            Override preconditions and postconditions behaviour for this call.
            If None, the default behaviour of this stage is used, as determined
            by the exraise constructor parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            is checked but before the application of the pipeline stage.
            Defaults to False.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if exraise is None:
            exraise = self._exraise
        if self._compound_prec(X):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            if self._is_fittable():
                if self.is_fitted:
                    res_df = self._transform(X, verbose=verbose)
                    if exraise and not self._compound_post(df=res_df):
                        self._raise_postcondition_error()
                    return res_df
                raise UnfittedPipelineStageError(
                    "transform of an unfitted pipeline stage was called!")
            res_df = self._transform(X, verbose=verbose)
            if exraise and not self._compound_post(df=res_df):
                self._raise_postcondition_error()
            return res_df
        if exraise:
            self._raise_precondition_error()
        return X

    def __add__(self, other):
        if isinstance(other, PdPipeline):
            return PdPipeline([self, *other._stages])
        if isinstance(other, PdPipelineStage):
            return PdPipeline([self, other])
        return NotImplemented

    def __str__(self):
        return f"PdPipelineStage: {self._desc}"

    def __repr__(self):
        return self.__str__()

    def description(self):
        """Returns the description of this pipeline stage"""
        return self._desc

    def _mem_str(self):
        total = asizeof(self)
        lines = []
        for a in dir(self):
            if not a.startswith('__'):
                att = getattr(self, a)
                if not callable(att):
                    size = asizeof(att)
                    if size > 500000:  # pragma: no cover
                        lines.append('  - {}, {:.2f}Mb ({:0>5.2f}%)\n'.format(
                            a, size / 1000000, 100 * size / total))
                    elif size > 1000:  # pragma: no cover
                        lines.append('  - {}, {:.2f}Kb ({:0>5.2f}%)\n'.format(
                            a, size / 1000, 100 * size / total))
                    else:
                        lines.append('  - {}, {}b ({:0>5.2f}%)\n'.format(
                            a, size, 100 * size / total))
        return ''.join(lines)


class ColumnsBasedPipelineStage(PdPipelineStage):
    """A pipeline stage that operates on a subset of dataframe columns.

    Parameters
    ---------
    columns : single label, iterable or callable
        The label, or an iterable of labels, of columns to use. Alternatively,
        this parameter can be assigned a callable returning an iterable of
        labels from an input pandas.DataFrame. See `pdpipe.cq`.
    exclude_columns : single label, iterable or callable, optional
        The label, or an iterable of labels, of columns to exclude, given the
        `columns` parameter. Alternatively, this parameter can be assigned a
        callable returning a labels iterable from an input pandas.DataFrame.
        See `pdpipe.cq`. Optional. By default no columns are excluded.
    desc_temp : str, optional
        If given, assumed to be a format string, and every appearance of {} in
        it is replaced with an appropriate string representation of the columns
        parameter, and is used as the pipeline description. Ignored if `desc`
        is provided.
    none_columns : iterable, callable or str, default 'error'
        Determines how None values supplied to the 'columns' parameter should
        be handled. If set to 'error', the default, a ValueError is raised if
        None is encountered. If set to 'all', it is interpreted to mean all
        columns of input dataframes should be operated on. If an iterable is
        provided it is interpreted as the default list of columns to operate on
        when `columns=None`. If a callable is provided, it is interpreted as
        the default column qualifier that determines input columns when
        `columns=None`.
    **kwargs
        Additionally supports all constructor parameters of PdPipelineStage.
    """

    @staticmethod
    def _interpret_columns_param(columns, none_error=False, none_columns=None):
        """Interprets the value provided to the columns parameter and returns
        a list version of it - if needed - a string representation of it.
        """
        if columns is None:
            if none_error:
                raise ValueError((
                    'None is not a valid argument for the columns parameter of'
                    ' this pipeline stage.'))
            return ColumnsBasedPipelineStage._interpret_columns_param(
                columns=none_columns)
        if isinstance(columns, str):
            # always check str first, because it has __iter__
            return [columns], columns
        if callable(columns):
            # if isinstance(columns, ColumnQualifier):
            #     return columns, columns.__repr__() or ''
            return columns, columns.__doc__ or ''
        # if it was a single string it was already made a list, and it's not a
        # callable, so it's either an iterable of labels... or
        if hasattr(columns, '__iter__'):
            return columns, ', '.join(str(elem) for elem in columns)
        # a single non-string label.
        return [columns], str(columns)

    def __init__(
            self, columns, exclude_columns=None, desc_temp=None,
            none_columns='error', **kwargs):
        self._exclude_columns = exclude_columns
        if exclude_columns:
            self._exclude_columns = self._interpret_columns_param(
                exclude_columns)
        self._none_error = False
        self._none_cols = None
        # handle none_columns
        if isinstance(none_columns, str):
            if none_columns == 'error':
                self._none_error = True
            elif none_columns == 'all':
                self._none_cols = AllColumns()
            else:
                raise ValueError((
                    "'error' and 'all' are the only valid string arguments"
                    " to the none_columns constructor parameter!"))
        elif hasattr(none_columns, '__iter__'):
            self._none_cols = none_columns
        elif callable(none_columns):
            self._none_cols = none_columns
        else:
            raise ValueError((
                "Valid arguments to the none_columns constructor parameter"
                " are 'error', 'all', an iterable of labels or a callable!"
            ))
        # done handling none_columns
        self._col_arg, self._col_str = self._interpret_columns_param(
            columns, self._none_error, none_columns=self._none_cols)
        if (kwargs.get('desc') is None) and desc_temp:
            kwargs['desc'] = desc_temp.format(self._col_str)
        if kwargs.get('exmsg') is None:
            kwargs['exmsg'] = (
                'Pipeline stage failed because not all columns {} '
                'were found in the input dataframe.'
            ).format(self._col_str)
        super().__init__(**kwargs)

    def _is_fittable(self):
        return is_fittable_column_qualifier(self._col_arg)

    @staticmethod
    def __get_cols_by_arg(col_arg, df, fit=False):
        try:
            if fit:
                # try to treat col_arg as a fittable column qualifier
                return col_arg.fit_transform(df)
            # else, no need to fit, so try to treat _col_arg as a callable
            return col_arg(df)
        except AttributeError:
            # got here cause col_arg has no fit_transform method...
            try:
                # so try and treat it as a callable again
                return col_arg(df)
            except TypeError:
                # calling col_arg 2 lines above failed; its a list of labels
                return col_arg
        except TypeError:
            # calling _col_arg 10 lines above failed; its a list of labels
            return col_arg

    def _get_columns(self, df, fit=False):
        cols = ColumnsBasedPipelineStage.__get_cols_by_arg(
            self._col_arg, df, fit=fit)
        if self._exclude_columns:
            exc_cols = ColumnsBasedPipelineStage.__get_cols_by_arg(
                self._exclude_columns, df, fit=fit)
            return [x for x in cols if x not in exc_cols]
        return cols

    def _prec(self, df):
        return set(self._get_columns(df=df)).issubset(df.columns)

    @abc.abstractmethod
    def _transformation(self, df, verbose, fit):
        raise NotImplementedError((
            "Classes extending ColumnsBasedPipelineStage must implement the "
            "_transformation method!"))

    def _fit_transform(self, df, verbose):
        self.is_fitted = True
        return self._transformation(df, verbose, fit=True)

    def _transform(self, df, verbose):
        return self._transformation(df, verbose, fit=False)


def _always_true(x):
    return True


class AdHocStage(PdPipelineStage):
    """An ad-hoc stage of a pandas DataFrame-processing pipeline.

    The signature for both the `transform` and the optional `fit_transform`
    callables is adaptive: The first argument is used positionally (so no
    specific name is assumed or used) to supply the callable with the pandas
    DataFrame object to transform. The following additional keyword arguments
    are supplied if the are included in the callable's signature:
    `verbose` - Passed on from PdPipelineStage's `fit`, `fit_transform`
    and `apply` methods.

    `fit_context` and `application_context` - Provides fit-specific and
    application-specific contexts (see `PdpApplicationContext`) usually
    available to pipeline stages using `self.fit_context` and
    `self.application_context`.

    Parameters
    ----------
    transform : callable
        The transformation this stage applies to dataframes. If the
        fit_transform parameter is also populated than this transformation is
        only applied on calls to transform. See documentation for the exact
        signature.
    fit_transform : callable, optional
        The transformation this stage applies to dataframes, only on
        fit_transform. Optional. See documentation for the exact signature.
    prec : callable, default None
        A callable that returns a boolean value. Represent a a precondition
        used to determine whether this stage can be applied to a given
        dataframe. If None is given, set to a function always returning True.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1, 'a'], [2, 'b']], [1, 2], ['num', 'char'])
        >>> drop_num = pdp.AdHocStage(
        ...   transform=lambda df: df.drop(['num'], axis=1),
        ...   prec=lambda df: 'num' in df.columns
        ... )
        >>> drop_num.apply(df)
          char
        1    a
        2    b
    """

    def __init__(self, transform, fit_transform=None, prec=None, **kwargs):
        if prec is None:
            prec = _always_true
        self._adhoc_transform = transform
        self._adhoc_fit_transform = fit_transform
        self._adhoc_prec = prec
        self._transform_kwargs = _get_args_list(self._adhoc_transform)
        try:
            self._fit_transform_kwargs = _get_args_list(
                self._adhoc_fit_transform)
        except TypeError:  # fit_transform is None
            self._fit_transform_kwargs = {}
        super().__init__(**kwargs)

    def _prec(self, df):
        return self._adhoc_prec(df)

    def _fit_transform(self, df, verbose):
        self.is_fitted = True
        if self._adhoc_fit_transform is None:
            self.is_fitted = True
            return self._transform(df, verbose)
        kwargs = {
            'verbose': verbose,
            'fit_context': self.fit_context,
            'application_context': self.application_context,
        }
        kwargs = {
            k: v for k, v in kwargs.items() if k in self._fit_transform_kwargs}
        return self._adhoc_fit_transform(df, **kwargs)

    def _transform(self, df, verbose):
        kwargs = {
            'verbose': verbose,
            'fit_context': self.fit_context,
            'application_context': self.application_context,
        }
        kwargs = {
            k: v for k, v in kwargs.items() if k in self._transform_kwargs}
        return self._adhoc_transform(df, **kwargs)


class PdPipeline(PdPipelineStage, collections.abc.Sequence):
    """A pipeline for processing pandas DataFrame objects.

    `transformer_getter` is useful to avoid applying pipeline stages that are
    aimed to filter out items in a big dataset to create a training set for a
    machine learning model, for example, but should not be applied on future
    individual items to be transformed by the fitted pipeline.

    Parameters
    ----------
    stages : list
        A list of PdPipelineStage objects making up this pipeline.
    transform_getter : callable, optional
        A callable that can be applied to the fitted pipeline to produce a
        sub-pipeline of it which should be used to transform dataframes after
        the pipeline has been fitted. If not given, the fitted pipeline is used
        entirely.
    """

    _DEF_EXC_MSG = 'Pipeline precondition failed!'

    def __init__(self, stages, transformer_getter=None, **kwargs):
        self._stages = stages
        self._trans_getter = transformer_getter
        self.is_fitted = False
        super_kwargs = {
            'exraise': False,
            'exmsg': PdPipeline._DEF_EXC_MSG,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    # implementing a collections.abc.Sequence abstract method
    def __getitem__(self, index):
        if isinstance(index, slice):
            return PdPipeline(self._stages[index])

        if isinstance(index, list) and all(isinstance(x, str) for x in index):
            stages = [stage for stage in self._stages if stage._name in index]
            return PdPipeline(stages)

        if isinstance(index, str):
            stages = [stage for stage in self._stages if stage._name == index]
            if len(stages) == 0:
                raise ValueError(f"'{index}' is not exist.")
            return stages[0]

        return self._stages[index]

    # implementing a collections.abc.Sequence abstract method
    def __len__(self):
        return len(self._stages)

    def _prec(self, df):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _post(self, df):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _transform(self, df, verbose):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _post_transform_lock(self):
        self.application_context.lock()
        self.fit_context.lock()

    def apply(self, df, exraise=None, verbose=False, time=False):
        """Applies this pipeline stage to the given dataframe.

        If the stage is not fitted fit_transform is called. Otherwise,
        transform is called.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to which this pipeline stage will be applied.
        exraise : bool, default None
            Determines behaviour if the precondition of composing stages is not
            fulfilled by the input dataframe: If True, a
            pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If not given, or set to None, the default behaviour of
            each stage is used, as determined by its 'exraise' constructor
            parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            is checked but before the application of the pipeline stage.
            Defaults to False.
        time : bool, default False
            If True, per-stage application time is measured and reported when
            pipeline application is done.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        self.application_context = PdpApplicationContext()
        if self.is_fitted:
            res = self.transform(
                X=df,
                exraise=exraise,
                verbose=verbose,
                time=time
            )
            self._post_transform_lock()
            return res
        self.fit_context = PdpApplicationContext()
        res = self.fit_transform(
            X=df,
            exraise=exraise,
            verbose=verbose,
            time=time
        )
        self._post_transform_lock()
        return res

    def __timed_fit_transform(self, X, y=None, exraise=None, verbose=None):
        self.application_context = PdpApplicationContext()
        self.fit_context = PdpApplicationContext()
        inter_x = X
        times = []
        prev = time.time()
        for i, stage in enumerate(self._stages):
            try:
                stage.fit_context = self.fit_context
                stage.application_context = self.application_context
                inter_x = stage.fit_transform(
                    X=inter_x,
                    y=None,
                    exraise=exraise,
                    verbose=verbose,
                )
                now = time.time()
                times.append(now - prev)
                prev = now
            except Exception as e:
                raise PipelineApplicationError(
                    f"Exception raised in stage [ {i}] {stage}"
                ) from e
        self.is_fitted = True
        print("\nPipeline total application time: {:.3f}s.\n Details:".format(
            sum(times)))
        print(self.__times_str__(times))
        self._post_transform_lock()
        return inter_x

    def fit_transform(self, X, y=None, exraise=None, verbose=None, time=False):
        """Fits this pipeline and transforms the input dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to transform and fit this pipeline by.
        y : array-like, optional
            Targets for supervised learning.
        exraise : bool, default None
            Determines behaviour if the precondition of composing stages is not
            fulfilled by the input dataframe: If True, a
            pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If not given, or set to None, the default behaviour of
            each stage is used, as determined by its 'exraise' constructor
            parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            of each stage is checked but before its application. Otherwise, no
            messages are printed.
        time : bool, default False
            If True, per-stage application time is measured and reported when
            pipeline application is done.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if time:
            return self.__timed_fit_transform(
                X=X, y=y, exraise=exraise, verbose=verbose)
        inter_x = X
        self.application_context = PdpApplicationContext()
        self.fit_context = PdpApplicationContext()
        for i, stage in enumerate(self._stages):
            try:
                stage.fit_context = self.fit_context
                stage.application_context = self.application_context
                inter_x = stage.fit_transform(
                    X=inter_x,
                    y=None,
                    exraise=exraise,
                    verbose=verbose,
                )
            except Exception as e:
                raise PipelineApplicationError(
                    f"Exception raised in stage [ {i}] {stage}"
                ) from e
        self._post_transform_lock()
        self.is_fitted = True
        return inter_x

    def fit(self, X, y=None, exraise=None, verbose=None, time=None):
        """Fits this pipeline without transforming the input dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to fit this pipeline by.
        y : array-like, optional
            Targets for supervised learning.
        exraise : bool, default None
            Determines behaviour if the precondition of composing stages is not
            fulfilled by the input dataframe: If True, a
            pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If not given, or set to None, the default behaviour of
            each stage is used, as determined by its 'exraise' constructor
            parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            of each stage is checked but before its application. Otherwise, no
            messages are printed.
        time : bool, default False
            If True, per-stage application time is measured and reported when
            pipeline application is done.

        Returns
        -------
        pandas.DataFrame
            The input dataframe, unchanged.
        """
        self.fit_transform(
            X=X,
            y=None,
            exraise=exraise,
            verbose=verbose,
            time=time,
        )
        return X

    def __timed_transform(self, X, y=None, exraise=None, verbose=None):
        inter_x = X
        times = []
        prev = time.time()
        self.application_context = PdpApplicationContext()
        self.fit_context = PdpApplicationContext()
        for i, stage in enumerate(self._stages):
            try:
                stage.fit_context = self.fit_context
                stage.application_context = self.application_context
                inter_x = stage.transform(
                    X=inter_x,
                    y=None,
                    exraise=exraise,
                    verbose=verbose,
                )
                now = time.time()
                times.append(now - prev)
                prev = now
            except Exception as e:
                raise PipelineApplicationError(
                    f"Exception raised in stage [ {i}] {stage}"
                ) from e
        self.is_fitted = True
        print("\nPipeline total application time: {:.3f}s.\n Details:".format(
            sum(times)))
        print(self.__times_str__(times))
        self._post_transform_lock()
        return inter_x

    def transform(self, X, y=None, exraise=None, verbose=None, time=False):
        """Transforms the given dataframe without fitting this pipeline.

        If any stage in this pipeline is fittable but is not fitted, an
        UnfittedPipelineStageError is raised before transformation starts.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to transform.
        y : array-like, optional
            Targets for supervised learning.
        exraise : bool, default None
            Determines behaviour if the precondition of composing stages is not
            fulfilled by the input dataframe: If True, a
            pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If not given, or set to None, the default behaviour of
            each stage is used, as determined by its 'exraise' constructor
            parameter.
        verbose : bool, default False
            If True an explanation message is printed after the precondition
            of each stage is checked but before its application. Otherwise, no
            messages are printed.
        time : bool, default False
            If True, per-stage application time is measured and reported when
            pipeline application is done.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        for stage in self._stages:
            if stage._is_fittable() and not stage.is_fitted:
                raise UnfittedPipelineStageError((
                    "PipelineStage {} in pipeline is fittable but"
                    " unfitted!").format(stage))
        if time:
            return self.__timed_transform(
                X=X, y=y, exraise=exraise, verbose=verbose)
        inter_df = X
        self.application_context = PdpApplicationContext()
        for i, stage in enumerate(self._stages):
            try:
                stage.application_context = self.application_context
                inter_df = stage.transform(
                    X=inter_df,
                    y=None,
                    exraise=exraise,
                    verbose=verbose,
                )
            except Exception as e:
                raise PipelineApplicationError(
                    f"Exception raised in stage [ {i}] {stage}"
                ) from e
        self._post_transform_lock()
        return inter_df

    __call__ = apply

    def __add__(self, other):
        if isinstance(other, PdPipeline):
            return PdPipeline([*self._stages, *other._stages])
        if isinstance(other, PdPipelineStage):
            return PdPipeline([*self._stages, other])
        return NotImplemented

    def __times_str__(self, times):
        res = "A pdpipe pipeline:\n"
        stime = sum(times)
        if stime > 0:  # pragma: no cover
            percentages = [100 * x / stime for x in times]
        else:  # pragma: no cover
            percentages = [0 for x in times]
        res += '[ 0] [{:0>5.2f}s ({:0>5.2f}%)]  '.format(
            times[0], percentages[0]
        ) + "\n      ".join(
            textwrap.wrap(self._stages[0].description())
        ) + '\n'
        for i, stage in enumerate(self._stages[1:]):
            res += '[{:>2}] [{:0>5.2f}s ({:0>5.2f}%)]  '.format(
                i + 1, times[i + 1], percentages[i + 1]
            ) + "\n      ".join(
                textwrap.wrap(stage.description())
            ) + '\n'
        return res

    def __str__(self):
        res = "A pdpipe pipeline:\n"
        res += '[ 0]  ' + "\n      ".join(
            textwrap.wrap(self._stages[0].description())) + '\n'
        for i, stage in enumerate(self._stages[1:]):
            res += '[{:>2}]  '.format(i + 1) + "\n      ".join(
                textwrap.wrap(stage.description())) + '\n'
        return res

    def _mem_str(self, total):
        total = asizeof(self)
        lines = []
        for i, stage in enumerate(self._stages):
            size = asizeof(stage)
            if size > 500000:  # pragma: no cover
                lines.append('[{:>2}] {:.2f}Mb ({:0>5.2f}%), {}\n'.format(
                    i, size / 1000000, 100 * size / total,
                    stage.description()))
            elif size > 1000:  # pragma: no cover
                lines.append('[{:>2}] {:.2f}Kb ({:0>5.2f}%), {}\n'.format(
                    i, size / 1000, 100 * size / total, stage.description()))
            else:
                lines.append('[{:>2}] {:}b ({:0>5.2f}%), {}\n'.format(
                    i, size, 100 * size / total, stage.description()))
            lines.append(stage._mem_str())
        return ''.join(lines)

    def memory_report(self):
        """Prints a detailed memory report of the pipeline object to screen.

        To get better memory estimates make sure the pympler Python package is
        installed. Without it, sys.getsizeof is used, which can be extremely
        underestimate memory size of Python objects.
        """
        print("=== Pipeline memory report ===")
        size = asizeof(self)
        if size > 500000:  # pragma: no cover
            print("Total pipeline size in memory: {:.2f}Mb".format(
                size / 1000000))
        elif size > 1000:  # pragma: no cover
            print("Total pipeline size in memory: {:.2f}Kb".format(
                size / 1000))
        else:
            print("Total pipeline size in memory: {:.2f}b".format(
                size))
        print("Per-stage memory structure:")
        print(self._mem_str(total=size))

    def get_transformer(self):
        """Return the transformer induced by this fitted pipeline.

           This transformer is a `pdpipe` pipeline that transforms input data
           in a way corresponding to this pipline after it has been fitted. By
           default this is the pipeline itself, but the `transform_getter`
           constructor parameter can be used to return a sub-pipeline of the
           fitted pipeline instead, for cases where some stages should only be
           applied when fitting this pipeline to data.

        Returns
        -------
        pdpipe.PdPipeline
            The corresponding transformer pipeline induced by this pipeline.
        """
        try:
            return self._trans_getter(self)
        except TypeError:  # pragma: no cover
            return self

    # def drop(self, index):
    #     """Returns this pipeline with the stage of the given index removed.
    #     Arguments
    #     ---------
    #     index


def make_pdpipeline(*stages):
    """Constructs a PdPipeline from the given pipeline stages.

    Parameters
    ----------
    *stages : pdpipe.PipelineStage objects
       PdPipeline stages given as positional arguments.

    Returns
    -------
    p : pdpipe.PdPipeline
        The resulting pipeline.

    Examples
    --------
        >>> import pdpipe as pdp
        >>> p = make_pdpipeline(pdp.ColDrop('count'), pdp.DropDuplicates())
    """
    return PdPipeline(stages=stages)
