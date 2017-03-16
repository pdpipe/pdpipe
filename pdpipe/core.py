"""Defines pipelines for processing Pandas.DataFrame-based datasets.

>>> import pdpipe as pdp
>>> pipeline = pdp.ColDrop('Name') + pdp.Bin({'Speed': [0,5]})
>>> pipeline = pdp.ColDrop('Name').bin({'Speed': [0,5]}, drop=True)
"""

import sys
import inspect
import types
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
        " to this pipline stage.")
    return doc.replace(first_line, new_first_line, 1)


def __load_stage_attribute__(class_obj):

    def _append_stage_func(self, *args, **kwds):
        # self is always a PipelineStage
        return self + class_obj(*args, **kwds)
    _append_stage_func.__doc__ = __get_append_stage_attr_doc(class_obj)
    _append_stage_func.__name__ = class_obj.__name__.lower()
    _append_stage_func.__signature__ = inspect.signature(class_obj.__init__)
    setattr(PipelineStage, class_obj.__name__.lower(),
            _append_stage_func)

    unbound_method = types.MethodType(_append_stage_func, class_obj)
    setattr(class_obj, class_obj.__name__.lower(),
            unbound_method)


def __load_stage_attributes_from_module__(module_name):
    module_obj = sys.modules[module_name]
    for name, obj in inspect.getmembers(module_obj):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            class_obj = getattr(module_obj, name)
            if issubclass(class_obj, PipelineStage) and (
                    class_obj.__name__ != 'PipelineStage'):
                __load_stage_attribute__(class_obj)


__STAGES_SUBMODULES = ['pdpipe.core', 'pdpipe.basic_stages']

def __load_stage_attributes__():
    for module_name in __STAGES_SUBMODULES:
        __load_stage_attributes_from_module__(module_name)


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
    _DEF_APPLY_MSG = 'Applying a pipline stage...'
    _DEF_DESCRIPTION = 'A pipline stage.'

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
            The daraframe to which this pipeline stage will be applied.
        exraise : bool, default None
            Determines behaviour if the precondition of this stage is not
            fulfilled by the given dataframe: If True,
            a pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If None, the default behaviour of this stage is used, as
            determined by the exraise constructor parameter.
        verbose : bool, default False
            If True an explaination message is printed after the precondition
            is checker but before the application of the pipeline stage.
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
                msg = '- '+ '\n  '.join(textwrap.wrap(self._appmsg))
                print(msg, flush=True)
            return self._op(df, verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
        return df

    __call__ = apply

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


class AdHocStage(PipelineStage):
    """An ad-hoc stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    op : callable
        The operation this stage applies to dataframes.
    prec : callable, default None
        A callable that returns a boolean value. Represneta a precondition
        used to determine whether this stage can be applied to a given
        dataframe. If None is given, set to a function always returning True.
    """

    def __init__(self, op, prec=None, **kwargs):
        if prec is None:
            prec = lambda x: True
        self._adhoc_op = op
        self._adhoc_prec = prec
        super().__init__(**kwargs)

    def _prec(self, df):
        return self._adhoc_prec(df)

    def _op(self, df, verbose):
        return self._adhoc_op(df)


class Pipeline(PipelineStage, collections.abc.Sequence):
    """A pipeline for processing pandas DataFrame objects.

    Parameters
    ----------
    stages : list
        A list of PipelineStage objects making up this pipeline.
    """

    _DEF_EXC_MSG = 'Pipeline precondition failed!'
    _DEF_APP_MSG = 'Applying a pipline...'

    def __init__(self, stages, **kwargs):
        self._stages = stages
        super_kwargs = {
            'exraise': False,
            'exmsg' : Pipeline._DEF_EXC_MSG,
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
        # pass
        prev_df = df
        for stage in self._stages:
            prev_df = stage.apply(prev_df, exraise, verbose)
        return prev_df

    __call__ = apply

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline([*self._stages, *other._stages])
        elif isinstance(other, PipelineStage):
            return Pipeline([*self._stages, other])
        else:
            return NotImplemented

    def __str__(self):
        res = "A pdpipe pipline:\n"
        res += '[ 0]  ' +  "\n      ".join(
            textwrap.wrap(self._stages[0].__str__())) + '\n'
        for i, stage in enumerate(self._stages[1:]):
            res += '[{:>2}]  '.format(i+1) + "\n      ".join(
                textwrap.wrap(stage.__str__())) + '\n'
        return res
