"""Defines pipelines for processing pandas.DataFrame-based datasets."""

import re
import sys
import abc
import time
import inspect
import collections
import textwrap
from typing import Tuple, Union, Iterable, Optional

import numpy
import pandas
import pandas as pd

from .run_time_parameters import DynamicParameter

try:
    from pympler.asizeof import asizeof
except ImportError:
    from sys import getsizeof as asizeof

from .cfg import (
    LOAD_STAGE_ATTRIBUTES,
)
from .cq import is_fittable_column_qualifier, AllColumns
from .shared import (
    POS_ARG_MISMTCH_PAT,
    _get_args_list,
    _always_true,
)
from .exceptions import (
    FailedPreconditionError,
    FailedPostconditionError,
    UnfittedPipelineStageError,
    PipelineApplicationError,
)


# === loading stage attributes ===


def __get_append_stage_attr_doc(class_obj: object) -> str:
    doc = class_obj.__doc__
    if doc is None:  # pragma: no cover
        return
    first_line = doc[0 : doc.find(".") + 1]  # noqa: E203
    if "An" in first_line:
        new_first_line = first_line.replace("An", "Create and adds an", 1)
    else:
        new_first_line = first_line.replace("A", "Create and adds a", 1)
    new_first_line = new_first_line[0:-1] + (" to this pipeline stage.")
    return doc.replace(first_line, new_first_line, 1)


def __load_stage_attribute__(class_obj: object) -> None:
    def _append_stage_func(self, *args, **kwds):
        # self is always a PdPipelineStage
        return self + class_obj(*args, **kwds)

    _append_stage_func.__doc__ = __get_append_stage_attr_doc(class_obj)
    _append_stage_func.__name__ = class_obj.__name__  # .lower()
    _append_stage_func.__signature__ = inspect.signature(class_obj.__init__)
    setattr(PdPipelineStage, class_obj.__name__, _append_stage_func)

    # unbound_method = types.MethodType(_append_stage_func, class_obj)
    # setattr(class_obj, class_obj.__name__, unbound_method)


def __load_stage_attributes_from_module__(module_name: object) -> None:
    if not LOAD_STAGE_ATTRIBUTES:
        return  # pragma: no cover
    module_obj = sys.modules[module_name]
    for name, obj in inspect.getmembers(module_obj):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            class_obj = getattr(module_obj, name)
            if issubclass(class_obj, PdPipelineStage) and (
                class_obj.__name__ != "PdPipelineStage"
            ):
                __load_stage_attribute__(class_obj)


# === basic classes ===


class PdpApplicationContext:
    """
    An object encapsulating the application context of a pipeline.

    It is meant to communicate data, information and variables between
    different stages of a pipeline during its application.

    Parameters
    ----------
    fit_context : PdpApplicationContext, optional
        Another application context object, representing the application
        context of a previous fit of the pipelline this application context
        is initialized for. Optional.
    """

    def __init__(
        self,
        fit_context: Optional["PdpApplicationContext"] = None,
    ) -> None:
        self._locked = False
        self._fit_context = fit_context
        self._dict = {}

    def __getitem__(self, key: object) -> object:
        return self._dict[key]

    def __setitem__(self, key: object, value: object) -> None:
        if not self._locked:
            self._dict[key] = value

    def __delitem__(self, key: object) -> object:
        if not self._locked:
            self._dict.__delitem__(key)

    def get(self, key: object, default: object = None) -> object:
        """
        Return the value for key if key is in the dictionary, else default.

        If default is not given, it defaults to None, so that this method never
        raises a KeyError.

        Parameters
        ----------
        key : object
            The key of the mapping to get.
        default : object
            The the key is not found, this value is returned instead.

        Returns
        -------
        object
            The value mapped to the given key in this dictionary. If it is not
            mapped, then default is returned. If default was not provided,
            None is returned.
        """
        return self._dict.get(key, default)

    def items(self) -> object:
        """
        Return a new view of the context’s items ((key, value) pairs).

        Returns
        -------
        object
            A new view of the context’s items ((key, value) pairs).
        """
        return self._dict.items()

    def keys(self):
        """
        Return a new view of the context's keys.

        Returns
        -------
        object
            A new view of the context's keys.
        """
        return self._dict.keys()

    def pop(self, key: object, default: object) -> object:
        """
        Remove the given key from this dict and return its mapped value.

        If key is in the dictionary, remove it and return its value, else
        return default. If default is not given and key is not in the
        dictionary, a KeyError is raised.

        Parameters
        ----------
        key : object
            The key of the mapping to get.
        default : object
            The the key is not found, this value is returned instead.

        Returns
        -------
        object
            The value mapped to the given key in this dictionary. If it is not
            mapped, then default is returned. If default was not provided,
            a KeyError is raised.
        """
        if not self._locked:
            return self._dict.pop(key, default)
        return self._dict.__getitem__(key)

    def clear(self) -> None:
        """Remove all items from the dictionary."""
        if not self._locked:
            self._dict.clear()

    def popitem(self) -> object:
        """Not implemented."""
        raise NotImplementedError

    def update(self, other: dict) -> None:
        """
        Update self with key-value pairs from another dict.

        This overwrite any existing mappings with keys that are also
        mapped in other.

        update() accepts either another dictionary object or an iterable of
        key/value pairs (as tuples or other iterables of length two). If
        keyword arguments are specified, the dictionary is then updated with
        those key/value pairs: d.update(red=1, blue=2).

        Parameters
        ----------
        other : dict
            The dict to get new key-value pairs from.
        """
        if not self._locked:
            self._dict.update(other)

    def lock(self) -> None:
        """Lock this application context for changes."""
        self._locked = True

    def fit_context(self) -> "PdpApplicationContext":
        """
        Return a locked PdpApplicationContext object of a previous fit.

        Returns
        -------
        PdpApplicationContext
            A locked PdpApplicationContext object of a previous fit.
        """
        return self._fit_context

    def is_locked(self) -> bool:
        """
        Return True if this application context is locked; False otherwise.

        Returns
        -------
        bool
            True if this application context is locked; False otherwise.
        """
        return self._locked


class AppContextMgr:
    """
    Manages application context objects over a specific pipeline application.

    Parameters
    ----------
    stage : PdPipelineStage
        The pipeline stage being applied.
    fit : bool
        If set to True, then this AppContextMgr is being used during a
        fit-transform of a pipeline; otherwise, it is being used during a
        transform. Optional, defaults to False.
    """

    def __init__(self, stage: "PdPipelineStage", fit: Optional[bool] = False):
        self.stage = stage
        self.fit = fit

    def __enter__(self):
        self.stage._is_being_applied = True
        if self.fit:
            self.stage._is_being_fitted = True
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stage._is_being_applied = False
        self.stage._is_being_fitted = False


class PdPipelineStage(abc.ABC):
    """
    A stage of a pandas DataFrame-processing pipeline.

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
    fit_context : PdpApplicationContext
        An application context object that is only re-initialized before
        `fit_transform` calls, and is locked after pipeline application. It is
        injected into the PipelineStage by the encapsulating pipeline object.
    application_context : PdpApplicationContext
        An application context object that is re-initialized before every
        pipeline application (so, also during transform operations of fitted
        pipelines), and is locked after pipeline application.It is injected
        into the PipelineStage by the encapsulating pipeline object.
    """

    _DEF_EXC_MSG = "Precondition failed in stage {}!"
    _DEF_DESCRIPTION = "A pipeline stage."
    _INIT_KWARGS = ["exraise", "exmsg", "desc", "prec", "skip", "name"]

    def __init__(
        self,
        exraise: Optional[bool] = True,
        exmsg: Optional[str] = None,
        desc: Optional[str] = None,
        prec: Optional[callable] = None,
        post: Optional[callable] = None,
        skip: Optional[callable] = None,
        name: Optional[str] = "",
    ) -> None:
        if not isinstance(name, str):
            raise ValueError(
                f"'name' must be a str, not {type(name).__name__}."
            )
        if desc is None:
            desc = PdPipelineStage._DEF_DESCRIPTION
        if exmsg is None:
            exmsg = PdPipelineStage._DEF_EXC_MSG.format(desc)

        # save input parameters
        self._exraise = exraise
        self._exmsg = exmsg
        self._exmsg_post = exmsg.replace(
            "precondition", "postcondition"
        ).replace("Precondition", "Postcondition")
        self._desc = desc
        self._prec_arg = prec
        self._post_arg = post
        self._skip = skip
        self._appmsg = f"{name + ': ' if name else ''}{desc}"
        self._name = name

        # inner stuff initializations
        self._is_an_Xy_transformer = False
        self._is_an_Xy_fit_transformer = False
        if hasattr(self.__class__, "_transform_Xy"):
            self._is_an_Xy_transformer = True
        if hasattr(self.__class__, "_fit_transform_Xy"):
            self._is_an_Xy_fit_transformer = True
        self.fit_context: PdpApplicationContext = None
        self.application_context: PdpApplicationContext = None
        self.is_fitted = False
        self._is_being_applied = False
        self._is_being_fitted = False
        self._dynamics = []  # list of parameters to be decided at runtime
        self._process_dynamics()

    def is_being_fitted_by_pipeline(self) -> bool:
        """
        Return True if this stage is being fitted, False otherwise.

        Returns
        -------
        bool
            True is this stage is being fitted; False otherwise.
        """
        try:
            return not self.fit_context.is_locked()
        except AttributeError:
            return False

    class_attrs = {
        "_exraise",
        "_exmsg",
        "_exmsg_post",
        "_desc",
        "_prec_arg",
        "_post_arg",
        "_skip",
        "_appmsg",
        "_name",
        "_is_an_Xy_transformer",
        "_is_an_Xy_fit_transformer",
        "fit_context",
        "application_context",
        "is_fitted",
        "_dynamics",
    }

    def _process_dynamics(self) -> None:
        """
        Create a list of Dynamic attributes of this stage.

        Returns
        -------
        None
        """
        potential_dynamics_attrs = set(self.__dict__).difference(
            self.class_attrs
        )
        for attr in potential_dynamics_attrs:
            attr_obj = self.__getattribute__(attr)
            if isinstance(attr_obj, DynamicParameter):
                self._dynamics.append({"name": attr, "callable": attr_obj})

    @classmethod
    def _init_kwargs(cls):
        return cls._INIT_KWARGS

    @classmethod
    def _split_kwargs(cls, kwargs: dict) -> Tuple[dict, dict]:
        """
        Split the given kwargs dict into init and non-init kwargs.

        Parameters
        ----------
        kwargs : dict
            The kwargs dict to split.

        Returns
        -------
        init_kwargs : dict
            The init kwargs dict.
        other_kwargs : dict
            The non-init kwargs dict.
        """
        init_kwargs = {
            k: v for k, v in kwargs.items() if k in cls._INIT_KWARGS
        }
        other_kwargs = {
            k: v for k, v in kwargs.items() if k not in cls._INIT_KWARGS
        }
        return init_kwargs, other_kwargs

    _MISSING_POS_ARG_PAT = re.compile(
        r"missing \d+ required positional argument"
    )

    @abc.abstractmethod
    def _prec(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
    ) -> bool:  # pylint: disable=W0613
        """
        Return True if this stage can be applied to the given dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The input dataframe.
        y : pandas.Series, optional
            A possible label column for processing of supervised learning
            datasets. Might also be inspected, if provided.

        Returns
        -------
        bool
            True if this stage this stage can be applied to the given
            dataframe; False otherwise.
        """
        raise NotImplementedError

    def _compound_prec(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
        fit: Optional[bool] = False,
    ) -> bool:
        """
        Return True if the input dataframe conforms to stage pre-condition.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to transform.
        y : pandas.Series, optional
            A possible label column for processing of supervised learning
            datasets. Might also be inspected, if provided.
        fit : bool
            If set to True, then this pre-condition check is understood to be
            performed during a fit-transform operation. Otherwise, it is
            assumed this is being performed during a transform operation.
            Optional. Set to False by default.

        Returns
        -------
        bool
            True if the input dataframe conforms to stage pre-condition; False
            otherwise.
        """
        if self._prec_arg:
            to_call = self._prec_arg
            if fit:
                try:
                    to_call = self._prec_arg.fit_transform
                except AttributeError:
                    pass
            try:
                return to_call(X, y)
            except TypeError as e:
                if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                    return to_call(X)
                raise e  # pragma: no cover

        # now, do the same for the _prec abstractmethod, so as to support
        # implementations only including X in their signature
        if y is None:
            try:
                return self._prec(X)
            except TypeError as e:
                if (
                    len(PdPipelineStage._MISSING_POS_ARG_PAT.findall(str(e)))
                    > 0
                ):
                    # self._prec is hopefully expecting y
                    return self._prec(X, y)
                raise e
        try:
            return self._prec(X, y)
        except TypeError as e:
            if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                return self._prec(X)
            raise e  # pragma: no cover

    def _post(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
    ) -> bool:  # pylint: disable=R0201,W0613
        """
        Return True if this stage resulted in an expected output frame.

        Parameters
        ----------
        X : pandas.DataFrame
            The transformed dataframe.
        y : pandas.Series, optional
            A possible label column for processing of supervised learning
            datasets. Might also be inspected, if provided.

        Returns
        -------
        bool
            True if this stage resulted in an expected output dataframe; False
            otherwise.
        """
        return True

    def _compound_post(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
        fit: Optional[bool] = False,
    ) -> bool:
        """
        An inner implementation of the post-condition functionality.

        Uses the constructor provided post-condition, if one was provided.
        Otherwise, uses the build-it function.
        """
        if self._post_arg:
            to_call = self._post_arg
            if fit:
                try:
                    to_call = self._post_arg.fit_transform
                except AttributeError:
                    pass
            try:
                return to_call(X, y)
            except TypeError as e:
                if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                    return to_call(X)
                raise e  # pragma: no cover

        # now, do the same for the _post abstractmethod, so as to support
        # implementations only including X in their signature
        if y is None:
            return self._post(X)
        try:
            return self._post(X, y)
        except TypeError as e:
            if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                return self._post(X)
            raise e  # pragma: no cover

    def _fit_transform(
        self,
        X: pandas.DataFrame,
        verbose: bool = False,
    ) -> pandas.DataFrame:
        """
        Fit this stage and transforms the input dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The input dataframe.
        verbose : bool, default False
            If True, might print informative messages.

        Returns
        -------
        pandas.DataFrame
            The transformed dataframe.
        """
        return self._transform(X, verbose=verbose)

    def _is_fittable(self) -> bool:
        if self.__class__._fit_transform == PdPipelineStage._fit_transform:
            return False
        return True

    def _raise_precondition_error(self) -> None:
        try:
            raise FailedPreconditionError(
                f"{self._exmsg} [Reason] {self._prec_arg._error_message}"
            )
        except AttributeError:
            raise FailedPreconditionError(self._exmsg)

    def _raise_postcondition_error(self) -> None:
        try:
            raise FailedPostconditionError(
                f"{self._exmsg_post} [Reason] {self._post_arg._error_message}"
            )
        except AttributeError:
            raise FailedPostconditionError(self._exmsg_post)

    @abc.abstractmethod
    def _transform(
        self,
        X: pandas.DataFrame,
        verbose: bool = False,
    ) -> pandas.DataFrame:
        """
        Transform an input dataframe without fitting this stage.

        Parameters
        ----------
        X : pandas.DataFrame
            The input dataframe.
        verbose : bool, optional
            If True, prints the progress of the transformation.

        Returns
        -------
        pandas.DataFrame
            The transformed dataframe.
        """
        raise NotImplementedError("_transform method not implemented!")

    @staticmethod
    def _cast_y_to_series(
        X: pandas.DataFrame,
        y: Union[pandas.Series, numpy.array, Iterable[object]],
    ) -> pandas.Series:
        """
        Cast the y labels input to a correclty-indexed pandas.Series.


        Parameters
        ----------
        X : pandas.DataFrame
            The input dataframe.
        y : pandas.Series, or numpy.ndarray, or Iterable[object]
            The label series/array.

        Returns
        -------
        pandas.Series
            The label array as a pandas.Series object.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length!")
        if isinstance(y, pandas.Series):
            post_y = pandas.Series(data=y.values, index=X.index)
        else:
            post_y = pandas.Series(data=y, index=X.index)
        return post_y

    @staticmethod
    def _align_Xy(
        X: pandas.DataFrame,
        y: pandas.Series,
        preX: pandas.DataFrame,
    ) -> Tuple[pandas.DataFrame, pandas.Series]:
        """
        Align the input dataframe and label series.

        The input dataframe and label series are assumed to have been indexed
        by the same index before a possible transformation to one of them,
        which might have dropped some rows/values in onf of them, and/or
        reorded the rows/values in one of them.

        Parameters
        ----------
        X : pandas.DataFrame
            The input dataframe.
        y : pandas.Series
            The label series.
        preX : pandas.DataFrame, optional
            The input dataframe before transformation.
        prey : pandas.Series, optional
            The label series before transformation.

        Returns
        -------
        pandas.DataFrame
            The aligned dataframe.
        pandas.Series
            The aligned label series.
        """
        # if values were dropped from y, reindex X according to y
        if len(y) < len(X):
            return X.loc[y.index], y
        try:
            # otherwise, transformation was almost certainly done on X (might
            # have been reordered), so reindex y according to X
            return X, y.loc[X.index]
        except KeyError:
            if (
                len(X) == len(preX)
                and not X.index.equals(preX.index)
                and (len(X) == len(y))
            ):
                # index values have changes, as in pandas.set_index
                post_y = y.copy()
                post_y.index = X.index
                return X, post_y
            return X, y

    def _should_skip(self, X, y) -> bool:
        if self._skip:
            try:
                return self._skip(X, y)
            except TypeError as e:
                if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                    return self._skip(X)
                else:
                    raise e  # pragma: no cover
        return False

    def apply(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = False,
    ) -> Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.Series]]:
        """
        Apply this pipeline stage to the given dataframe.

        If the stage is not fitted fit_transform is called. Otherwise,
        transform is called.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to which this pipeline stage will be applied.
        y : pandas.Series, optional
            A possible label column for processing of supervised learning
            datasets. Might also be transformed, if provided.
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
        pandas.DataFrame or Tuple[pandas.DataFrame, pandas.Series]
            The returned dataframe. If `y` was also provided, the transformed
            `X` and `y` are returned as a tuple instead.
        """
        if self.is_fitted:
            return self.transform(X, y=y, exraise=exraise, verbose=verbose)
        return self.fit_transform(X, y=y, exraise=exraise, verbose=verbose)

    __call__ = apply

    def fit_transform(self, X, y=None, exraise=None, verbose=False):
        """
        Fit this stage and transforms the given dataframe.

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
        with AppContextMgr(self, fit=True):
            if exraise is None:
                exraise = self._exraise
            if self._should_skip(X, y):
                if y is not None:
                    return X, y
                return X
            if y is not None:
                y = self._cast_y_to_series(X, y)
            if self._compound_prec(X, y, fit=True):
                if verbose:
                    msg = "- " + "\n  ".join(textwrap.wrap(self._appmsg))
                    print(msg, flush=True)
                if self._is_an_Xy_fit_transformer:
                    res_X, res_y = self._fit_transform_Xy(
                        X, y, verbose=verbose
                    )
                elif self._is_an_Xy_transformer:
                    res_X, res_y = self._transform_Xy(X, y, verbose=verbose)
                else:
                    res_X = self._fit_transform(X, verbose=verbose)
                    res_y = y
                self.is_fitted = True
                if exraise and not self._compound_post(
                    X=res_X, y=res_y, fit=True
                ):
                    self._raise_postcondition_error()
                if y is not None:
                    res_X, res_y = self._align_Xy(X=res_X, y=res_y, preX=X)
                    return res_X, res_y
                return res_X
            if exraise:
                self._raise_precondition_error()
            if y is not None:
                return X, y
            return X

    def fit(self, X, y=None, exraise=None, verbose=False):
        """
        Fit this stage without transforming the given dataframe.

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
        self.fit_transform(X, y=y, exraise=exraise, verbose=verbose)
        if y is not None:
            return X, y
        return X

    def transform(self, X, y=None, exraise=None, verbose=False):
        """
        Transform the given dataframe without fitting this stage.

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
        with AppContextMgr(self):
            if exraise is None:
                exraise = self._exraise
            if self._should_skip(X, y):
                if y is not None:
                    return X, y
                return X
            if y is not None:
                y = self._cast_y_to_series(X, y)
            if self._compound_prec(X, y):
                if verbose:
                    msg = "- " + "\n  ".join(textwrap.wrap(self._appmsg))
                    print(msg, flush=True)
                if self._is_fittable():
                    if self.is_fitted:
                        if self._is_an_Xy_transformer:
                            res_X, res_y = self._transform_Xy(
                                X, y, verbose=verbose
                            )
                        else:
                            res_X = self._transform(X, verbose=verbose)
                            res_y = y
                        if exraise and not self._compound_post(
                            X=res_X, y=res_y
                        ):
                            self._raise_postcondition_error()
                        if y is not None:
                            res_X, res_y = self._align_Xy(
                                X=res_X, y=res_y, preX=X
                            )
                            return res_X, res_y
                        return res_X
                    raise UnfittedPipelineStageError(
                        "transform of an unfitted pipeline stage was called!"
                    )
                if self._is_an_Xy_transformer:
                    res_X, res_y = self._transform_Xy(X, y, verbose=verbose)
                else:
                    res_X = self._transform(X, verbose=verbose)
                    res_y = y
                if exraise and not self._compound_post(X=res_X, y=res_y):
                    self._raise_postcondition_error()
                if y is not None:
                    res_X, res_y = self._align_Xy(X=res_X, y=res_y, preX=X)
                    return res_X, res_y
                return res_X
            if exraise:
                self._raise_precondition_error()
            # precondition doesn't hold, but we don't want to raise an error
            # as exraise is False, so return untransformed X or X, y
            if y is not None:
                return X, y
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

    def description(self) -> str:
        """
        Return the description of this pipeline stage.

        Returns
        -------
        str
            The description of this pipeline stage.
        """
        return self._desc

    def _mem_str(self):
        total = asizeof(self)
        lines = []
        for a in dir(self):
            if not a.startswith("__"):
                att = getattr(self, a)
                if not callable(att):
                    size = asizeof(att)
                    if size > 500000:  # pragma: no cover
                        lines.append(
                            "  - {}, {:.2f}Mb ({:0>5.2f}%)\n".format(
                                a, size / 1000000, 100 * size / total
                            )
                        )
                    elif size > 1000:  # pragma: no cover
                        lines.append(
                            "  - {}, {:.2f}Kb ({:0>5.2f}%)\n".format(
                                a, size / 1000, 100 * size / total
                            )
                        )
                    else:
                        lines.append(
                            "  - {}, {}b ({:0>5.2f}%)\n".format(
                                a, size, 100 * size / total
                            )
                        )
        return "".join(lines)


class ColumnsBasedPipelineStage(PdPipelineStage):
    """
    A pipeline stage that operates on a subset of dataframe columns.

    Parameters
    ----------
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

    _INIT_KWARGS = [
        "columns",
        "exclude_columns",
        "desc_temp",
        "none_columns",
    ] + PdPipelineStage._INIT_KWARGS

    @staticmethod
    def _interpret_columns_param(columns, none_error=False, none_columns=None):
        """Interprets the value provided to the columns parameter and returns
        a list version of it - if needed - and a string representation of it.
        """
        if columns is None:
            if none_error:
                raise ValueError(
                    (
                        "None is not a valid argument for the columns "
                        "parameter of this pipeline stage."
                    )
                )
            return ColumnsBasedPipelineStage._interpret_columns_param(
                columns=none_columns
            )
        if isinstance(columns, str):
            # always check str first, because it has __iter__
            return [columns], f"'{columns}'"
        if callable(columns):
            # if isinstance(columns, ColumnQualifier):
            #     return columns, columns.__repr__() or ''
            return columns, columns.__doc__ or ""
        # if it was a single string it was already made a list, and it's not a
        # callable, so it's either an iterable of labels... or
        if hasattr(columns, "__iter__"):
            return columns, ", ".join(f"'{str(elem)}'" for elem in columns)
        # a single non-string label.
        return [columns], f"'{columns}'"

    def __init__(
        self,
        columns,
        exclude_columns=None,
        desc_temp=None,
        none_columns="error",
        **kwargs,
    ):
        self._exclude_columns = exclude_columns
        if exclude_columns:
            (
                self._exclude_columns,
                self._exc_col_str,
            ) = self._interpret_columns_param(exclude_columns)
        self._none_error = False
        self._none_cols = None
        # handle none_columns
        if isinstance(none_columns, str):
            if none_columns == "error":
                self._none_error = True
            elif none_columns == "all":
                self._none_cols = AllColumns()
            else:
                raise ValueError(
                    (
                        "'error' and 'all' are the only valid string arguments"
                        " to the none_columns constructor parameter!"
                    )
                )
        elif hasattr(none_columns, "__iter__"):
            self._none_cols = none_columns
        elif callable(none_columns):
            self._none_cols = none_columns
        else:
            raise ValueError(
                (
                    "Valid arguments to the none_columns constructor parameter"
                    " are 'error', 'all', an iterable of labels or a callable!"
                )
            )
        # done handling none_columns
        self._col_arg, self._col_str = self._interpret_columns_param(
            columns, self._none_error, none_columns=self._none_cols
        )
        if exclude_columns:
            self._final_col_str = (
                f"{self._col_str} (except {self._exc_col_str})"
            )
        else:
            self._final_col_str = f"{self._col_str}"
        if (kwargs.get("desc") is None) and desc_temp:
            kwargs["desc"] = desc_temp.format(self._final_col_str)
        if kwargs.get("exmsg") is None:
            kwargs["exmsg"] = (
                "Pipeline stage failed because not all columns {} "
                "were found in the input dataframe."
            ).format(self._final_col_str)
        super().__init__(**kwargs)

    def _is_fittable(self):
        return is_fittable_column_qualifier(self._col_arg)

    @staticmethod
    def __get_cols_by_arg(col_arg, X, fit=False):
        try:
            if fit:
                # try to treat col_arg as a fittable column qualifier
                return col_arg.fit_transform(X)
            # else, no need to fit, so try to treat _col_arg as a callable
            return col_arg(X)
        except AttributeError:
            # got here cause col_arg has no fit_transform method...
            try:
                # so try and treat it as a callable again
                return col_arg(X)
            except TypeError:
                # calling col_arg 2 lines above failed; its a list of labels
                return col_arg
        except TypeError:
            # calling _col_arg 10 lines above failed; its a list of labels
            return col_arg

    def _get_columns(self, X, fit=False):
        cols = ColumnsBasedPipelineStage.__get_cols_by_arg(
            self._col_arg, X, fit=fit
        )
        if self._exclude_columns:
            exc_cols = ColumnsBasedPipelineStage.__get_cols_by_arg(
                self._exclude_columns, X, fit=fit
            )
            return [x for x in cols if x not in exc_cols]
        return cols

    def _prec(self, X, y=None):
        required_cols = set(self._get_columns(X, fit=self._is_being_fitted))
        return required_cols.issubset(X.columns)

    @abc.abstractmethod
    def _transformation(self, X, verbose, fit):
        raise NotImplementedError(
            (
                "Classes extending ColumnsBasedPipelineStage must implement "
                "the _transformation method!"
            )
        )

    def _fit_transform(self, X, verbose):
        self.is_fitted = True
        return self._transformation(X, verbose, fit=True)

    def _transform(self, X, verbose):
        return self._transformation(X, verbose, fit=False)


class AdHocStage(PdPipelineStage):
    """
    An ad-hoc stage of a pandas DataFrame-processing pipeline.

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
    **kwargs : object
        All PdPipelineStage constructor parameters are supported.

    Examples
    --------
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
                self._adhoc_fit_transform
            )
        except TypeError:  # fit_transform is None
            self._fit_transform_kwargs = {}
        super().__init__(**kwargs)

    def _prec(self, X, y=None):
        try:
            return self._adhoc_prec(X, y)
        except TypeError as e:
            if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                return self._adhoc_prec(X)
            raise e

    def _fit_transform(self, X, verbose):
        self.is_fitted = True
        if self._adhoc_fit_transform is None:
            self.is_fitted = True
            return self._transform(X, verbose)
        kwargs = {
            "verbose": verbose,
            "fit_context": self.fit_context,
            "application_context": self.application_context,
        }
        kwargs = {
            k: v for k, v in kwargs.items() if k in self._fit_transform_kwargs
        }
        return self._adhoc_fit_transform(X, **kwargs)

    def _transform(self, X, verbose):
        kwargs = {
            "verbose": verbose,
            "fit_context": self.fit_context,
            "application_context": self.application_context,
        }
        kwargs = {
            k: v for k, v in kwargs.items() if k in self._transform_kwargs
        }
        return self._adhoc_transform(X, **kwargs)


class PdPipeline(PdPipelineStage, collections.abc.Sequence):
    """
    A pipeline for processing pandas DataFrame objects.

    `transformer_getter` is useful to avoid applying pipeline stages that are
    aimed to filter out items in a big dataset to create a training set for a
    machine learning model, for example, but should not be applied on future
    individual items to be transformed by the fitted pipeline.

    Parameters
    ----------
    stages : list
        A list of PdPipelineStage objects making up this pipeline.
    transformer_getter : callable, optional
        A callable that can be applied to the fitted pipeline to produce a
        sub-pipeline of it which should be used to transform dataframes after
        the pipeline has been fitted. If not given, the fitted pipeline is used
        entirely.
    **kwargs : object
        All additional PdPipelineStage constructor parameters are supported.

    Attributes
    ----------
    fit_context : PdpApplicationContext
        An application context object that is only re-initialized before
        `fit_transform` calls, and is locked after pipeline application. It is
        injected into the PipelineStage by the encapsulating pipeline object.
    application_context : PdpApplicationContext
        An application context object that is re-initialized before every
        pipeline application (so, also during transform operations of fitted
        pipelines), and is locked after pipeline application.It is injected
        into the PipelineStage by the encapsulating pipeline object.
    is_fitted : bool
        Whether this pipeline has been fitted.
    """

    _DEF_EXC_MSG = "Pipeline precondition failed!"

    def __init__(self, stages, transformer_getter=None, **kwargs):
        self._stages = stages
        self._trans_getter = transformer_getter
        self.is_fitted = False
        super_kwargs = {
            "exraise": False,
            "exmsg": PdPipeline._DEF_EXC_MSG,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    # implementing a collections.abc.Sequence abstract method
    def __getitem__(self, index):
        stages = None
        if isinstance(index, slice):
            stages = self._stages[index]

        if isinstance(index, list) and all(isinstance(x, str) for x in index):
            stages = [stage for stage in self._stages if stage._name in index]

        if stages is not None:
            pline = PdPipeline(stages)
            pline.fit_context = self.fit_context
            pline.is_fitted = self.is_fitted
            return pline

        if isinstance(index, str):
            stages = [stage for stage in self._stages if stage._name == index]
            if len(stages) == 0:
                raise ValueError(f"'{index}' is not exist.")
            return stages[0]

        return self._stages[index]

    # implementing a collections.abc.Sequence abstract method
    def __len__(self):
        return len(self._stages)

    def _prec(self, X, y=None):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _post(self, X):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _transform(self, X, verbose):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _post_transform_lock(self):
        # Application context is discarded after pipeline application
        self.application_context = None
        self.fit_context.lock()

    @staticmethod
    def _use_dynamics(
        stage: PdPipelineStage,
        inter_X: pd.DataFrame,
        inter_y: Optional[pd.Series] = None,
    ) -> None:
        """
        Sets the dynamic parameter based on given callable

        Parameters
        ----------
        stage : pdpipe.PdPipelineStage
            The stage for which to provide the dynamic parameter.
        inter_X : pandas.DataFrame
            The input dataframe.
        inter_y : pandas.Series
            The input y labels. Optional

        Returns
        -------
        None
        """
        for dynamic in stage._dynamics:
            param = (
                dynamic["callable"](inter_X)
                if inter_y is None
                else dynamic["callable"](inter_X, inter_y)
            )
            setattr(stage, dynamic["name"], param)

    def apply(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = False,
        time: Optional[bool] = False,
        fit_context: Optional[dict] = {},
        application_context: Optional[dict] = {},
    ):
        """
        Apply this pipeline stage to the given dataframe.

        If the stage is not fitted fit_transform is called. Otherwise,
        transform is called.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe to which this pipeline stage will be applied.
        y : pandas.Series, optional
            A possible label column for processing of supervised learning
            datasets. Might also be transformed, if provided.
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
        fit_context : dict, option
            Context for the entire pipeline, is retained after the pipeline
            application is completed.
        application_context : dict, optional
            Context to add to the application context of this call. Can map
            str keys to arbitrary object values to be used by pipeline stages
            during this pipeline application. Discarded after pipeline
            application.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if self.is_fitted:
            res = self.transform(
                X,
                y,
                exraise=exraise,
                verbose=verbose,
                time=time,
                application_context=application_context,
            )
            return res
        res = self.fit_transform(
            X,
            y,
            exraise=exraise,
            verbose=verbose,
            time=time,
            fit_context=fit_context,
            application_context=application_context,
        )
        return res

    def __timed_fit_transform(
        self,
        X: pandas.DataFrame,
        y: Optional[Iterable] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = False,
        fit_context: Optional[dict] = {},
        application_context: Optional[dict] = {},
    ):
        self.fit_context = PdpApplicationContext()
        self.fit_context.update(fit_context)
        self.application_context = PdpApplicationContext()
        self.application_context.update(application_context)
        inter_X = X
        inter_y = y
        times = []
        prev = time.time()
        if y is None:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    inter_X = stage.fit_transform(
                        X=inter_X,
                        y=None,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                    now = time.time()
                    times.append(now - prev)
                    prev = now
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        else:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    inter_X, inter_y = stage.fit_transform(
                        X=inter_X,
                        y=inter_y,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                    now = time.time()
                    times.append(now - prev)
                    prev = now
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        self.is_fitted = True
        print(
            "\nPipeline total application time: {:.3f}s.\n Details:".format(
                sum(times)
            )
        )
        print(self.__times_str__(times))
        self._post_transform_lock()
        if y is None:
            return inter_X
        return inter_X, inter_y

    def fit_transform(
        self,
        X: pandas.DataFrame,
        y: Optional[Iterable] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = False,
        time: Optional[bool] = False,
        fit_context: Optional[dict] = {},
        application_context: Optional[dict] = {},
    ):
        """
        Fit this pipeline and transforms the input dataframe.

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
        fit_context : dict, option
            Context for the entire pipeline, is retained after the pipeline
            application is completed.
        application_context : dict, optional
            Context to add to the application context of this call. Can map
            str keys to arbitrary object values to be used by pipeline stages
            during this pipeline application. Discarded after pipeline
            application.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if time:
            return self.__timed_fit_transform(
                X,
                y,
                exraise=exraise,
                verbose=verbose,
                fit_context=fit_context,
                application_context=application_context,
            )
        inter_X = X
        inter_y = y
        self.application_context = PdpApplicationContext()
        self.application_context.update(application_context)
        self.fit_context = PdpApplicationContext()
        self.fit_context.update(fit_context)
        if y is None:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    self._use_dynamics(stage, inter_X, inter_y)

                    inter_X = stage.fit_transform(
                        X=inter_X,
                        y=None,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        else:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    self._use_dynamics(stage, inter_X, inter_y)

                    inter_X, inter_y = stage.fit_transform(
                        X=inter_X,
                        y=inter_y,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        self._post_transform_lock()
        self.is_fitted = True
        if y is None:
            return inter_X
        return inter_X, inter_y

    def fit(
        self,
        X: pandas.DataFrame,
        y: Optional[Iterable] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = False,
        time: Optional[bool] = False,
        fit_context: Optional[dict] = {},
        application_context: Optional[dict] = {},
    ):
        """
        Fit this pipeline without transforming the input dataframe.

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
        fit_context : dict, option
            Context for the entire pipeline, is retained after the pipeline
            application is completed.
        application_context : dict, optional
            Context to add to the application context of this call. Can map
            str keys to arbitrary object values to be used by pipeline stages
            during this pipeline application.

        Returns
        -------
        pandas.DataFrame
            The input dataframe, unchanged.
        """
        self.fit_transform(
            X,
            y,
            exraise=exraise,
            verbose=verbose,
            time=time,
            fit_context=fit_context,
            application_context=application_context,
        )
        if y is None:
            return X
        return X, y

    def __timed_transform(
        self,
        X: pandas.DataFrame,
        y: Optional[Iterable[float]] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = None,
        application_context: Optional[dict] = {},
    ) -> pandas.DataFrame:
        inter_X = X
        inter_y = y
        times = []
        prev = time.time()
        self.application_context = PdpApplicationContext()
        self.application_context.update(application_context)
        if y is None:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    inter_X = stage.transform(
                        X=inter_X,
                        y=None,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    now = time.time()
                    times.append(now - prev)
                    prev = now
                    stage.application_context = None
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        else:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    inter_X, inter_y = stage.transform(
                        X=inter_X,
                        y=inter_y,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                    now = time.time()
                    times.append(now - prev)
                    prev = now
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        self.is_fitted = True
        print(
            "\nPipeline total application time: {:.3f}s.\n Details:".format(
                sum(times)
            )
        )
        print(self.__times_str__(times))
        self._post_transform_lock()
        if y is None:
            return inter_X
        return inter_X, inter_y

    def transform(
        self,
        X: pandas.DataFrame,
        y: Optional[Iterable[float]] = None,
        exraise: Optional[bool] = None,
        verbose: Optional[bool] = None,
        time: Optional[bool] = False,
        application_context: Optional[dict] = {},
    ) -> pandas.DataFrame:
        """
        Transform the given dataframe without fitting this pipeline.

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
            pipeline application is performed.
        application_context : dict, optional
            Context to add to the application context of this call. Can map
            str keys to arbitrary object values to be used by pipeline stages
            during this pipeline application.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if time:
            return self.__timed_transform(
                X,
                y,
                exraise=exraise,
                verbose=verbose,
                application_context=application_context,
            )
        inter_X = X
        inter_y = y
        self.application_context = PdpApplicationContext()
        self.application_context.update(application_context)
        if y is None:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    self._use_dynamics(stage, inter_X, inter_y)

                    inter_X = stage.transform(
                        X=inter_X,
                        y=None,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        else:
            for i, stage in enumerate(self._stages):
                try:
                    stage.fit_context = self.fit_context
                    stage.application_context = self.application_context
                    self._use_dynamics(stage, inter_X, inter_y)

                    inter_X, inter_y = stage.transform(
                        X=inter_X,
                        y=inter_y,
                        exraise=exraise,
                        verbose=verbose,
                    )
                    stage.application_context = None
                except Exception as e:
                    stage.application_context = None
                    raise PipelineApplicationError(
                        f"Exception raised in stage [ {i}] {stage}"
                    ) from e
        self._post_transform_lock()
        if y is None:
            return inter_X
        return inter_X, inter_y

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
        res += (
            "[ 0] [{:0>5.2f}s ({:0>5.2f}%)]  ".format(times[0], percentages[0])
            + "\n      ".join(textwrap.wrap(self._stages[0].description()))
            + "\n"
        )
        for i, stage in enumerate(self._stages[1:]):
            res += (
                "[{:>2}] [{:0>5.2f}s ({:0>5.2f}%)]  ".format(
                    i + 1, times[i + 1], percentages[i + 1]
                )
                + "\n      ".join(textwrap.wrap(stage.description()))
                + "\n"
            )
        return res

    def __str__(self):
        res = "A pdpipe pipeline:\n"
        res += (
            "[ 0]  "
            + "\n      ".join(textwrap.wrap(self._stages[0].description()))
            + "\n"
        )
        for i, stage in enumerate(self._stages[1:]):
            res += (
                "[{:>2}]  ".format(i + 1)
                + "\n      ".join(textwrap.wrap(stage.description()))
                + "\n"
            )
        return res

    def _mem_str(self, total):
        total = asizeof(self)
        lines = []
        for i, stage in enumerate(self._stages):
            size = asizeof(stage)
            if size > 500000:  # pragma: no cover
                lines.append(
                    "[{:>2}] {:.2f}Mb ({:0>5.2f}%), {}\n".format(
                        i,
                        size / 1000000,
                        100 * size / total,
                        stage.description(),
                    )
                )
            elif size > 1000:  # pragma: no cover
                lines.append(
                    "[{:>2}] {:.2f}Kb ({:0>5.2f}%), {}\n".format(
                        i, size / 1000, 100 * size / total, stage.description()
                    )
                )
            else:
                lines.append(
                    "[{:>2}] {:}b ({:0>5.2f}%), {}\n".format(
                        i, size, 100 * size / total, stage.description()
                    )
                )
            lines.append(stage._mem_str())
        return "".join(lines)

    def memory_report(self):
        """
        Print a detailed memory report of the pipeline object to screen.

        To get better memory estimates make sure the pympler Python package is
        installed. Without it, sys.getsizeof is used, which can be extremely
        underestimate memory size of Python objects.
        """
        print("=== Pipeline memory report ===")
        size = asizeof(self)
        if size > 500000:  # pragma: no cover
            print(
                "Total pipeline size in memory: {:.2f}Mb".format(
                    size / 1000000
                )
            )
        elif size > 1000:  # pragma: no cover
            print(
                "Total pipeline size in memory: {:.2f}Kb".format(size / 1000)
            )
        else:
            print("Total pipeline size in memory: {:.2f}b".format(size))
        print("Per-stage memory structure:")
        print(self._mem_str(total=size))

    def get_transformer(self) -> "PdPipeline":
        """
        Return the transformer induced by this fitted pipeline.

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
    #     """Return this pipeline with the stage of the given index removed.
    #     Arguments
    #     ---------
    #     index


def make_pdpipeline(*stages: PdPipelineStage) -> PdPipeline:
    """
    Construct a PdPipeline from the given pipeline stages.

    This is a convenience method that wraps the PdPipeline constructor,
    essentially performing:
    return PdPipeline(stages=stages)

    Parameters
    ----------
    *stages : pdpipe.PipelineStage
        PdPipeline stages given as positional arguments.

    Returns
    -------
    pdpipe.PdPipeline
        The resulting pipeline.

    Examples
    --------
    >>> import pdpipe as pdp
    >>> p = make_pdpipeline(pdp.ColDrop('count'), pdp.DropDuplicates())
    """
    return PdPipeline(stages=stages)
