"""Defines pipelines for processing Pandas.DataFrame-based datasets.

>>> import pdpipe as pdp
>>> pipeline = pdp.ColDrop('Name') + pdp.Bin({'Speed': [0,5]})
>>> pipeline = pdp.ColDrop('Name').bin({'Speed': [0,5]}, drop=True)
"""

import sys
import inspect
import types


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


def __get_create_stage_attr_doc(class_obj):
    return class_obj.__doc__.replace("A", "Creates a", 1)


def __load_stage_attribute__(class_obj, pdp_module):

    def _append_stage_func(self, *args, **kwds):
        if isinstance(self, PipelineStage):
            return self + class_obj(*args, **kwds)
        return class_obj(self, *args, **kwds)
    _append_stage_func.__doc__ = __get_append_stage_attr_doc(class_obj)
    _append_stage_func.__name__ = class_obj.__name__.lower()
    _append_stage_func.__signature__ = inspect.signature(class_obj.__init__)
    setattr(PipelineStage, class_obj.__name__.lower(),
            _append_stage_func)

    unbound_method = types.MethodType(_append_stage_func, class_obj)
    setattr(class_obj, class_obj.__name__.lower(),
            unbound_method)

    # def _create_stage_func(*args, **kwds):
    #     return class_obj(*args, **kwds)
    # _create_stage_func.__doc__ = __get_create_stage_attr_doc(class_obj)
    # _create_stage_func.__name__ = class_obj.__name__.lower()
    # _create_stage_func.__signature__ = inspect.signature(class_obj.__init__)
    # setattr(pdp_module, class_obj.__name__.lower(),
    #         _create_stage_func)


def __load_stage_attributes_from_module__(module_name, pdp_module):
    module_obj = sys.modules[module_name]
    for name, obj in inspect.getmembers(module_obj):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            class_obj = getattr(module_obj, name)
            if issubclass(class_obj, PipelineStage) and (
                    class_obj.__name__ != 'PipelineStage'):
                __load_stage_attribute__(class_obj, pdp_module)


__STAGES_SUBMODULES = ['pdpipe.core', 'pdpipe.basic_stages']

def __load_stage_attributes__(pdp_module):
    for module_name in __STAGES_SUBMODULES:
        __load_stage_attributes_from_module__(module_name, pdp_module)


# === basic classes

class FailedPreconditionError(Exception):
    """An exception raised when a pipeline stage is applied to a dataframe for
    which the stage precondition does not hold.
    """
    pass


class PipelineStage(object):
    """A stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    exraise : bool, optional
        If true, a pdpipe.FailedPreconditionError is raised when this
        stage is applied to a dataframe for which the precondition does
        not hold. Otherwise the stage is skipped. Defaults to True.
    exmsg : str, optional
        The message of the exception that is raised on a failed
        precondition if exraise is set to True. A default message is used
        if none is given.
    appmsg : str, optional
        The message printed when this stage is applied with verbose=True.
        A default message is used if none is given.
    """

    DEF_EXC_MSG = 'Precondition failed!'
    DEF_APPLY_MSG = 'Applying a pipline stage...'

    def __init__(self, exraise=True, exmsg=None, appmsg=None):
        if exmsg is None:
            exmsg = PipelineStage.DEF_EXC_MSG
        if appmsg is None:
            appmsg = PipelineStage.DEF_APPLY_MSG
        self._exraise = exraise
        self._exmsg = exmsg
        self._appmsg = appmsg

    def _prec(self, df):  # pylint: disable=R0201,W0613
        """Returns True if this stage can be applied to the given dataframe."""
        return True

    def _op(self, df, verbose):
        """The operation to apply to dataframes passed through this stage."""
        return df

    def apply(self, df, exraise=None, verbose=False):
        """Applies this pipeline stage to the given dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The daraframe to which this pipeline stage will be applied.
        exraise : bool, optional
            Determines behaviour if the precondition of this stage is not
            fulfilled by the given dataframe: If True,
            pdpipe.FailedPreconditionError is raised. If False, the stage is
            skipped. If not given, the default behaviour of this stage is used,
            as determined by the exraise constructor parameter.
        verbose : bool
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
                print(self._appmsg, flush=True)
            return self._op(df, verbose)
        if exraise:
            raise FailedPreconditionError(self._exmsg)
        return df

    __call__ = apply

    def __add__(self, other):
        if isinstance(other, PipelineStage):
            return Pipeline([self, other])
        elif isinstance(other, Pipeline):
            return Pipeline([self, *other.stages])
        else:
            return NotImplemented

    def __str__(self):
        return "Do nothing"

    def __repr__(self):
        return self.__str__()


class AdHocStage(PipelineStage):
    """An ad-hoc stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    op : callable
        The operation this stage applies to dataframes.
    prec : callable, optional
        A callable that returns a boolean value. Represneta a precondition
        used to determine whether this stage can be applied to a given
        dataframe. Defaults to a function always returning True.
    description: str, optional
        A description of this pipeline stage. Used in string
        representations of this pipline stage and of any pipeline
        containing it. Defaults to a non-informative description.
    exraise : boolean, optional
        If true, an exception is raised when this stage is applies to a
        dataframe for which the precondition does not hold. Defaults to
        True.
    exmsg : str, optional
        The message of the exception that is raised on a failed
        precondition if exraise is set to True.
    appmsg : str, optional
        The message printed when this stage is applied with verbose=True.
        A default message is used if none is given.
    """

    DEF_DESC = "An ad-hoc pipline stage"

    def __init__(self, op, prec=lambda x: True, description=DEF_DESC,
                 exraise=True, exmsg=None, appmsg=None):
        self._op = op
        self._prec = prec
        self._desc = description
        super(AdHocStage, self).__init__(
            exraise=exraise, exmsg=exmsg, appmsg=appmsg)

    def __str__(self):
        return self._desc


class Pipeline(PipelineStage):
    """A pipeline for processing pandas DataFrame objects.

    Parameters
    ----------
    stages : list
        A list of PipelineStage objects making up this pipeline.
    appmsg : str, optional
        The message printed when this pipeline is applied with verbose=True.
        A default message is used if none is given.
    """

    DEF_APPLY_MSG = 'Applying a pipline...'

    def __init__(self, stages, appmsg=None):
        if appmsg is None:
            appmsg = self.DEF_APPLY_MSG
        super(Pipeline, self).__init__(exraise=False, appmsg=appmsg)
        self._stages = stages

    def _prec(self, df):
        return True

    def _op(self, df, verbose):
        pass  # Pipeline overrides apply in a way which makes this moot

    def apply(self, df, exraise=None, verbose=False):
        prev_df = df
        for stage in self._stages:
            prev_df = stage.apply(df=prev_df, exraise=exraise, verbose=verbose)
        return prev_df


    def __add__(self, other):
        if isinstance(other, PipelineStage):
            return Pipeline([*self._stages, other])
        elif isinstance(other, Pipeline):
            return Pipeline([*self._stages, *other.stages])
        else:
            return NotImplemented

    def __str__(self):
        res = "A pdpipe pipline:\n"
        res += 'df -> ' + self._stages[0].__str__() + '\n'
        for stage in self._stages[1:]:
            res += '   -> ' + stage.__str__() + '\n'
        return res
