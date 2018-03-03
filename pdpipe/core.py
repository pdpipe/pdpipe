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
        # self is always a PipelineStage
        return self + class_obj(*args, **kwds)
    _append_stage_func.__doc__ = __get_append_stage_attr_doc(class_obj)
    _append_stage_func.__name__ = class_obj.__name__  # .lower()
    _append_stage_func.__signature__ = inspect.signature(class_obj.__init__)
    setattr(PipelineStage, class_obj.__name__, _append_stage_func)

    # unbound_method = types.MethodType(_append_stage_func, class_obj)
    # setattr(class_obj, class_obj.__name__, unbound_method)


def __load_stage_attributes_from_module__(module_name):
    module_obj = sys.modules[module_name]
    for name, obj in inspect.getmembers(module_obj):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            class_obj = getattr(module_obj, name)
            if issubclass(class_obj, PipelineStage) and (
                    class_obj.__name__ != 'PipelineStage'):
                __load_stage_attribute__(class_obj)


# === basic classes ===

class FailedPreconditionError(Exception):
    """An exception raised when a pipeline stage is applied to a dataframe for
    which the stage precondition does not hold.
    """
    pass


class PipelineStage(abc.ABC):
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
            exmsg = PipelineStage._DEF_EXC_MSG
        if appmsg is None:
            appmsg = PipelineStage._DEF_APPLY_MSG
        if desc is None:
            desc = PipelineStage._DEF_DESCRIPTION
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

    @abc.abstractmethod
    def _op(self, df, verbose):
        """The operation to apply to dataframes passed through this stage."""
        raise NotImplementedError

    def apply(self, df, exraise=None, verbose=False):
        """Applies this pipeline stage to the given dataframe.

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
                return self._transform(df, verbose)
            return self._op(df, verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
        return df

    __call__ = apply

    def fit_transform(self, df, exraise=None, verbose=False):
        """Transform the given dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to be transformed.
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
            return self._op(df, verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
        return df

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline([self, *other._stages])
        elif isinstance(other, PipelineStage):
            return Pipeline([self, other])
        else:
            return NotImplemented

    def __str__(self):
        return self._desc

    def __repr__(self):
        return self.__str__()


def _always_true(x):
    return True


class AdHocStage(PipelineStage):
    """An ad-hoc stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    op : callable
        The operation this stage applies to dataframes.
    prec : callable, default None
        A callable that returns a boolean value. Represent a a precondition
        used to determine whether this stage can be applied to a given
        dataframe. If None is given, set to a function always returning True.
    """

    def __init__(self, op, prec=None, **kwargs):
        if prec is None:
            prec = _always_true
        self._adhoc_op = op
        self._adhoc_prec = prec
        super().__init__(**kwargs)

    def _prec(self, df):
        return self._adhoc_prec(df)

    def _op(self, df, verbose):
        return self._adhoc_op(df)


class Pipeline(PipelineStage, collections.abc.Sequence):
    """A pipeline for processing pandas DataFrame objects.

    transformer_getter is usefull to avoid applying pipeline stages that are
    aimed to filter out items in a big dataset to create a training set for a
    machine learning model, for example, but should not be applied on future
    individual items to be transformed by the fitted pipeline.

    Parameters
    ----------
    stages : list
        A list of PipelineStage objects making up this pipeline.
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
            'exmsg': Pipeline._DEF_EXC_MSG,
            'appmsg': Pipeline._DEF_APP_MSG
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    # implementing a collections.abc.Sequence abstract method
    def __getitem__(self, index):
        if isinstance(index, slice):
            return Pipeline(self._stages[index])
        return self._stages[index]

    # implementing a collections.abc.Sequence abstract method
    def __len__(self):
        return len(self._stages)

    def _prec(self, df):
        # Pipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def _op(self, df, verbose):
        # Pipeline overrides apply in a way which makes this moot
        raise NotImplementedError

    def apply(self, df, exraise=None, verbose=False):
        inter_df = df
        for stage in self._stages:
            inter_df = stage.apply(inter_df, exraise, verbose)
        return inter_df

    def fit_transform(self, df, exraise=None, verbose=None):
        inter_df = df
        for stage in self._stages:
            inter_df = stage.fit_transform(inter_df, exraise, verbose)
        return inter_df

    __call__ = apply

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline([*self._stages, *other._stages])
        elif isinstance(other, PipelineStage):
            return Pipeline([*self._stages, other])
        else:
            return NotImplemented

    def __str__(self):
        res = "A pdpipe pipeline:\n"
        res += '[ 0]  ' + "\n      ".join(
            textwrap.wrap(self._stages[0].__str__())) + '\n'
        for i, stage in enumerate(self._stages[1:]):
            res += '[{:>2}]  '.format(i+1) + "\n      ".join(
                textwrap.wrap(stage.__str__())) + '\n'
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
