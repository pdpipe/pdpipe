"""Defines pipelines for processing Pandas.DataFrame-based datasets.

>>> import pdpipe as pdp
>>> pipeline = pdp.ColDrop('Name') + pdp.Bin({'Speed': [0,5]})
>>> pipeline = pdp.ColDrop('Name').Bin({'Speed': [0,5]}, drop=True)
"""

import sys
import inspect
import abc
import collections
import textwrap

from .exceptions import (
    FailedPreconditionError,
    UnfittedPipelineStageError,
)


# === loading stage attributes ===

def __get_append_stage_attr_doc(class_obj):
    doc = class_obj.__doc__
    first_line = doc[0:doc.find('.')+1]
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


class PdPipelineStage(abc.ABC):
    """A stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    exraise : bool, default True
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped.
    exmsg : str, default None
        The message of the exception that is raised on a failed
        precondition if exraise is set to True. A default message is used
        if None is given.
    appmsg : str, default None
        The message printed when this stage is applied with verbose=True.
        A default message is used if None is given.
    desc : str, default None
        A short description of this stage, used as its string representation.
        A default description is used if None is given.
    """

    _DEF_EXC_MSG = 'Precondition failed!'
    _DEF_APPLY_MSG = 'Applying a pipeline stage...'
    _DEF_DESCRIPTION = 'A pipeline stage.'
    _INIT_KWARGS = ['exraise', 'exmsg', 'appmsg', 'desc']

    def __init__(self, exraise=True, exmsg=None, appmsg=None, desc=None):
        if exmsg is None:
            exmsg = PdPipelineStage._DEF_EXC_MSG
        if appmsg is None:
            appmsg = PdPipelineStage._DEF_APPLY_MSG
        if desc is None:
            desc = PdPipelineStage._DEF_DESCRIPTION
        self._exraise = exraise
        self._exmsg = exmsg
        self._appmsg = appmsg
        self._desc = desc
        self.is_fitted = False

    @classmethod
    def _init_kwargs(cls):
        return cls._INIT_KWARGS

    @abc.abstractmethod
    def _prec(self, df):  # pylint: disable=R0201,W0613
        """Returns True if this stage can be applied to the given dataframe."""
        raise NotImplementedError

    def _fit_transform(self, df, verbose):
        """Fits this stage and transforms the input dataframe."""
        return self._transform(df, verbose)

    def _is_fittable(self):
        if self.__class__._fit_transform == PdPipelineStage._fit_transform:
            return False
        return True

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
            Determines behaviour if the precondition of this stage is not
            fulfilled by the given dataframe: If True,
            a pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If None, the default behaviour of this stage is used, as
            determined by the exraise constructor parameter.
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
        if self._prec(df):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            if self.is_fitted:
                return self._transform(df, verbose=verbose)
            return self._fit_transform(df, verbose=verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
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
            Determines behaviour if the precondition of this stage is not
            fulfilled by the given dataframe: If True,
            a pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If None, the default behaviour of this stage is used, as
            determined by the exraise constructor parameter.
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
        if self._prec(X):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            return self._fit_transform(X, verbose=verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
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
            Determines behaviour if the precondition of this stage is not
            fulfilled by the given dataframe: If True,
            a pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If None, the default behaviour of this stage is used, as
            determined by the exraise constructor parameter.
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
        if self._prec(X):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            self._fit_transform(X, verbose=verbose)
            return X
        if exraise:
            raise FailedPreconditionError(self._exmsg)
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
            Determines behaviour if the precondition of this stage is not
            fulfilled by the given dataframe: If True,
            a pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If None, the default behaviour of this stage is used, as
            determined by the exraise constructor parameter.
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
        if self._prec(X):
            if verbose:
                msg = '- ' + '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            if self._is_fittable():
                if self.is_fitted:
                    return self._transform(X, verbose=verbose)
                raise UnfittedPipelineStageError(
                    "transform of an unfitted pipeline stage was called!")
            return self._transform(X, verbose=verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
        return X

    def __add__(self, other):
        if isinstance(other, PdPipeline):
            return PdPipeline([self, *other._stages])
        elif isinstance(other, PdPipelineStage):
            return PdPipeline([self, other])
        else:
            return NotImplemented

    def __str__(self):
        return "PdPipelineStage: {}".format(self._desc)

    def __repr__(self):
        return self.__str__()

    def description(self):
        """Returns the description of this pipeline stage"""
        return self._desc


def _always_true(x):
    return True


class AdHocStage(PdPipelineStage):
    """An ad-hoc stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    transform : callable
        The transformation this stage applies to dataframes.
    prec : callable, default None
        A callable that returns a boolean value. Represent a a precondition
        used to determine whether this stage can be applied to a given
        dataframe. If None is given, set to a function always returning True.
    """

    def __init__(self, transform, prec=None, **kwargs):
        if prec is None:
            prec = _always_true
        self._adhoc_transform = transform
        self._adhoc_prec = prec
        super().__init__(**kwargs)

    def _prec(self, df):
        return self._adhoc_prec(df)

    def _transform(self, df, verbose):
        try:
            return self._adhoc_transform(df, verbose=verbose)
        except TypeError:
            return self._adhoc_transform(df)


class PdPipeline(PdPipelineStage, collections.abc.Sequence):
    """A pipeline for processing pandas DataFrame objects.

    transformer_getter is usefull to avoid applying pipeline stages that are
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
    _DEF_APP_MSG = 'Applying a pipeline...'

    def __init__(self, stages, transformer_getter=None, **kwargs):
        self._stages = stages
        self._trans_getter = transformer_getter
        self.is_fitted = False
        super_kwargs = {
            'exraise': False,
            'exmsg': PdPipeline._DEF_EXC_MSG,
            'appmsg': PdPipeline._DEF_APP_MSG
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    # implementing a collections.abc.Sequence abstract method
    def __getitem__(self, index):
        if isinstance(index, slice):
            return PdPipeline(self._stages[index])
        return self._stages[index]

    # implementing a collections.abc.Sequence abstract method
    def __len__(self):
        return len(self._stages)

    def _prec(self, df):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _transform(self, df, verbose):
        # PdPipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def apply(self, df, exraise=None, verbose=False):
        inter_df = df
        for stage in self._stages:
            inter_df = stage.apply(inter_df, exraise, verbose)
        return inter_df

    def fit_transform(self, X, y=None, exraise=None, verbose=None):
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

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        inter_x = X
        for stage in self._stages:
            inter_x = stage.fit_transform(
                X=inter_x,
                y=None,
                exraise=exraise,
                verbose=verbose,
            )
        return inter_x

    def fit(self, X, y=None, exraise=None, verbose=None):
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
        )
        return X

    def transform(self, X, y=None, exraise=None, verbose=None):
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
        inter_df = X
        for stage in self._stages:
            inter_df = stage.transform(
                X=inter_df,
                y=None,
                exraise=exraise,
                verbose=verbose,
            )
        return inter_df

    __call__ = apply

    def __add__(self, other):
        if isinstance(other, PdPipeline):
            return PdPipeline([*self._stages, *other._stages])
        elif isinstance(other, PdPipelineStage):
            return PdPipeline([*self._stages, other])
        else:
            return NotImplemented

    def __str__(self):
        res = "A pdpipe pipeline:\n"
        res += '[ 0]  ' + "\n      ".join(
            textwrap.wrap(self._stages[0].description())) + '\n'
        for i, stage in enumerate(self._stages[1:]):
            res += '[{:>2}]  '.format(i+1) + "\n      ".join(
                textwrap.wrap(stage.description())) + '\n'
        return res

    def get_transformer(self):
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
    import pdpipe as pdp
    make_pdpipeline(pdp.ColDrop('a'), pdp.Bin('speed'))
    """
    return PdPipeline(stages=stages)
